"""Pluginfer REST API (FastAPI).

Lazy on purpose: importing a lightweight submodule — notably
``api.governance_gateway``, the standalone ON-PREMISES governance +
token-thrift suite — must NOT drag in ``api.main``'s full mesh + torch
dependency chain (~1650 modules, ~3 s). ``build_app`` stays importable
via ``from api import build_app``; it just loads on first access
instead of at package import. This is what lets an organization deploy
the governance gateway with zero mesh/torch on their premises.
"""

__all__ = ["build_app"]


def __getattr__(name):
    if name == "build_app":
        from .main import build_app
        return build_app
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
