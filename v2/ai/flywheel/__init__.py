"""Self-improvement flywheel: collect inference + outcome pairs, label,
fine-tune, evaluate, promote.

The collector + evaluator are real; labeler + fine_tuner are honest
stubs (they need on-chain receipt parsing and a long-running training
budget which are out of session scope).
"""

from .collector import FlywheelCollector, InferenceLogEvent
from .labeler import OutcomeLabeler, LabelingNotImplementedError
from .fine_tuner import WeeklyFineTuner, FineTuningNotImplementedError
from .evaluator import ABEvaluator

__all__ = [
    "FlywheelCollector",
    "InferenceLogEvent",
    "OutcomeLabeler",
    "LabelingNotImplementedError",
    "WeeklyFineTuner",
    "FineTuningNotImplementedError",
    "ABEvaluator",
]
