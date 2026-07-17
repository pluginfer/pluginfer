"""Outcome labeler - honest stub.

The job: take collected inference events and pair each with the on-chain
outcome (job completed? in expected runtime? at expected cost? was the
predicted price within bounds of clearing?) so the next fine-tuning run
has supervised labels.

This requires:
  - Parsing on-chain task_receipt transactions and matching them by
    request_id (added to the receipt's metadata field)
  - Querying the auction-clearing log for actual clear price
  - Reading the provider's reputation delta produced after audit

None of those are 1-hour work; the surface is here so the flywheel
pipeline can be wired end-to-end with an honest stub today and the
real implementation can drop in later without changing callers.
"""

from __future__ import annotations


class LabelingNotImplementedError(NotImplementedError):
    pass


class OutcomeLabeler:
    def __init__(self, *args, **kwargs) -> None:
        raise LabelingNotImplementedError(
            "OutcomeLabeler requires on-chain task_receipt + auction-clearing "
            "log integration. The collector is real and the labeler can drop "
            "in once the receipts carry the inference request_id metadata "
            "field. Roadmap: TODO §7 CP-AI-FINAL+ flywheel deep-wire."
        )
