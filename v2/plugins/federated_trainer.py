"""
Federated Trainer Plugin — REAL DiLoCo Worker
=============================================
This is the plugin that workers expose to the network. One invocation =
one DiLoCo inner-loop round on this node's local data.

Replaces the previous version which contained `time.sleep` in place of
training and returned a dummy gradient. This version actually:

    * builds a real PyTorch model from a serializable spec,
    * loads the aggregator's global weights (with SHA-256 verification),
    * runs K real SGD/AdamW steps on locally-resident data,
    * computes a real parameter delta,
    * 8-bit quantizes the delta for WAN transport,
    * returns a SHA-256-checked, schema-validated payload.

Local data never leaves the worker. Only the gradient delta is shipped.

Wire format
-----------
input_data:
    {
        'task': 'diloco_round',          # only mode supported here
        'model_spec':       {...},       # see core/diloco_models.py
        'global_weights':   <base64 str> | None,   # serialized state_dict
        'inner_steps':      int,         # default 50
        'inner_lr':         float,       # default 1e-2
        'batch_size':       int,         # default 32
        'optimizer':        'sgd' | 'adamw',
        'quantize':         bool,        # default True
        'data':             {            # local data spec (see below)
            'kind': 'inline_tensors',    # or 'synthetic_regression', 'synthetic_classify'
            'x_b64': '...', 'y_b64': '...',
            'shape_x': [...], 'shape_y': [...],
            'dtype_x': 'float32', 'dtype_y': 'float32' | 'int64',
        },
        'audit_seed':       int | None,  # if set, deterministic so aggregator can verify
        'device':           'auto' | 'cpu' | 'cuda' | 'mps',
    }

return value:
    {
        'status':              'success',
        'quantized_delta_b64': str,
        'metrics':             {...},
        'base_weights_hash':   str,      # what we actually trained from
        'final_weights_hash':  str,      # so the aggregator can audit
        'compression_ratio':   float,
        'device':              str,
        'param_count':         int,
    }
"""

from typing import Any, Dict
import base64
import logging
import math
import time

from core.plugin_base import PluginBase

logger = logging.getLogger(__name__)


# Module-level worker cache: building a model is expensive, so we keep
# one per (arch, init_seed) tuple. Each worker process will hold ~one.
_WORKER_CACHE: Dict[str, Any] = {}


def _decode_b64(s: str | None) -> bytes | None:
    if not s:
        return None
    return base64.b64decode(s)


def _build_local_data(spec: Dict[str, Any]):
    """
    Materialize this worker's local training data. The plugin contract
    is that whoever invokes this plugin already has data on this box;
    the spec just tells us *how* to construct it.

    Three kinds:
      'inline_tensors'        : caller passed base64 fp32 tensors
      'synthetic_regression'  : reproducible y = Wx + b + noise
      'synthetic_classify'    : reproducible 2-class linearly separable

    Synthetic modes exist mainly for testing — production replaces this
    branch with on-device data loaders (PyTorch DataLoader pointing at
    the user's local dataset).
    """
    import torch
    kind = spec.get("kind", "synthetic_regression")

    if kind == "inline_tensors":
        x = torch.frombuffer(bytearray(_decode_b64(spec["x_b64"])),
                             dtype=getattr(torch, spec.get("dtype_x", "float32"))
                             ).reshape(spec["shape_x"]).clone()
        y = torch.frombuffer(bytearray(_decode_b64(spec["y_b64"])),
                             dtype=getattr(torch, spec.get("dtype_y", "float32"))
                             ).reshape(spec["shape_y"]).clone()
        return x, y

    if kind == "synthetic_regression":
        seed = int(spec.get("seed", 0))
        n = int(spec.get("n", 1024))
        d = int(spec.get("d", 16))
        noise = float(spec.get("noise", 0.05))
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, d, generator=g)
        # Use a fixed "ground truth" so all workers see the same task.
        gt = torch.Generator().manual_seed(424242)
        W = torch.randn(d, 1, generator=gt)
        b = torch.randn(1, generator=gt)
        y = (x @ W + b) + noise * torch.randn(n, 1, generator=g)
        return x, y

    if kind == "synthetic_classify":
        seed = int(spec.get("seed", 0))
        n = int(spec.get("n", 1024))
        d = int(spec.get("d", 16))
        g = torch.Generator().manual_seed(seed)
        gt = torch.Generator().manual_seed(424243)
        W = torch.randn(d, generator=gt)
        x = torch.randn(n, d, generator=g)
        logits = x @ W
        y = (logits > 0).long()
        return x, y

    raise ValueError(f"Unknown data kind: {kind}")


class FederatedTrainer(PluginBase):
    def config(self) -> Dict[str, Any]:
        return {
            "name": "federated_trainer",
            "version": "3.0.0",
            "description": (
                "Real DiLoCo worker: runs K-step inner loop locally, "
                "returns 8-bit quantized parameter delta. Local data "
                "never leaves the node."
            ),
            "category": "ai_training",
            "tags": ["training", "diloco", "federated", "privacy", "torch"],
            "cost_per_exec": 0.05,
            "inputs": {
                "model_spec": "dict",
                "global_weights": "base64-str|None",
                "inner_steps": "int",
                "inner_lr": "float",
                "batch_size": "int",
                "optimizer": "str",
                "quantize": "bool",
                "data": "dict",
                "audit_seed": "int|None",
                "device": "str",
            },
            "outputs": {
                "quantized_delta_b64": "base64-str",
                "metrics": "dict",
                "base_weights_hash": "str",
                "final_weights_hash": "str",
                "compression_ratio": "float",
                "device": "str",
                "param_count": "int",
            },
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Local imports keep startup cheap when this plugin isn't called.
        try:
            import torch  # noqa: F401
        except ImportError:
            return {"error": "PyTorch not installed on this node"}

        from core.diloco_worker import DiLoCoWorker, InnerLoopConfig, make_tensor_iter

        try:
            self.validate_input(input_data, ["model_spec", "data"])

            model_spec = input_data["model_spec"]
            cfg = InnerLoopConfig(
                inner_steps=int(input_data.get("inner_steps", 50)),
                inner_lr=float(input_data.get("inner_lr", 1e-2)),
                inner_momentum=float(input_data.get("inner_momentum", 0.9)),
                weight_decay=float(input_data.get("weight_decay", 0.0)),
                batch_size=int(input_data.get("batch_size", 32)),
                grad_clip_norm=input_data.get("grad_clip_norm", 1.0),
                optimizer=str(input_data.get("optimizer", "sgd")),
                quantize=bool(input_data.get("quantize", True)),
            )
            device_pref = str(input_data.get("device", "auto"))

            cache_key = f"{model_spec.get('arch')}|{model_spec.get('init_seed', 0)}|{device_pref}"
            worker = _WORKER_CACHE.get(cache_key)
            if worker is None:
                worker = DiLoCoWorker(model_spec=model_spec, device_pref=device_pref)
                _WORKER_CACHE[cache_key] = worker

            # Materialize local data (never goes on the wire).
            x, y = _build_local_data(input_data["data"])
            audit_seed = input_data.get("audit_seed")
            iter_seed = int(audit_seed) if audit_seed is not None else int(time.time() * 1e3) & 0x7FFFFFFF
            data_iter = make_tensor_iter(x, y, seed=iter_seed)

            # Decode global weights and run.
            global_payload = _decode_b64(input_data.get("global_weights"))
            t0 = time.time()
            result = worker.run_round(
                data_iter=data_iter,
                cfg=cfg,
                global_weights_payload=global_payload,
            )
            wall = time.time() - t0

            # Sanity: training should not produce NaNs.
            if math.isnan(result.final_loss) or math.isinf(result.final_loss):
                return {"error": "Loss diverged (NaN/Inf). Reduce inner_lr."}

            return {
                "status": "success",
                "quantized_delta_b64": base64.b64encode(result.quantized_delta).decode("ascii"),
                "metrics": {
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                    "examples_seen": result.examples_seen,
                    "inner_steps": result.inner_steps,
                    "inner_wall_time": result.wall_time,
                    "plugin_wall_time": wall,
                    "delta_norm": result.delta_norm,
                },
                "base_weights_hash": result.base_weights_hash,
                "final_weights_hash": result.final_weights_hash,
                "compression_ratio": result.compression_ratio,
                "device": result.device,
                "param_count": int(result.metadata.get("param_count", 0)),
                "audit": {
                    "seed": iter_seed if audit_seed is not None else None,
                },
            }

        except ValueError as ve:
            return {"error": f"Bad input: {ve}", "code": "INVALID_INPUT"}
        except Exception as ex:  # pragma: no cover — surface to client
            logger.exception("federated_trainer failed")
            return {"error": str(ex), "code": "EXECUTION_ERROR"}
