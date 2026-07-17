"""Pluginfer Python SDK.

Minimal usage:

    from pluginfer import Pluginfer
    client = Pluginfer(api_key="pf_live_...", base_url="https://api.pluginfer.network")
    job = client.jobs.submit(kind="llm.completion", payload={"prompt": "hi"})
    print(job.state)
    for evt in client.jobs.stream(job.job_id):
        print(evt)
    result = client.jobs.result(job.job_id)
    print(result.result_b64)

Authentication: pass `api_key=` for Bearer token auth, or call
`client.auth.login_with_wallet(wallet)` for ECDSA challenge-response.
"""

from .client import Pluginfer, PluginferClient
from .fine_tune import (
    FineTuneError,
    FineTuneSpec,
    TrainingCheckpoint,
    TrainingJob,
    fine_tune,
    fine_tune_blocking,
)
from .exceptions import (
    AuthenticationError,
    InsufficientBalanceError,
    JobNotFoundError,
    PluginferError,
    RateLimitError,
)
from .types import (
    Job,
    JobResult,
    JobState,
    Provider,
    Status,
    WalletBalance,
)

__all__ = [
    "AuthenticationError",
    "FineTuneError",
    "FineTuneSpec",
    "InsufficientBalanceError",
    "Job",
    "JobNotFoundError",
    "JobResult",
    "JobState",
    "Pluginfer",
    "PluginferClient",
    "PluginferError",
    "Provider",
    "RateLimitError",
    "Status",
    "TrainingCheckpoint",
    "TrainingJob",
    "WalletBalance",
    "fine_tune",
    "fine_tune_blocking",
]

__version__ = "1.0.0"
