"""SmartContractVM regression test (post-W22 hardened sandbox).

Pre-W22 the test used implicit `storage` globals; the hardened sandbox
bans dunder access + most builtins so contracts now receive `storage`
and `msg` as explicit kwargs and return `(value, new_storage)` to
mutate state. This file pins the post-W22 contract.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

from core.smart_contracts import SmartContractVM  # noqa: E402


COUNTER_CODE = """
def increment(val, storage, msg):
    new_count = storage.get('count', 0) + val
    return (new_count, {'count': new_count})

def get_count(storage, msg):
    return storage.get('count', 0)
"""


@pytest.fixture()
def storage_path(tmp_path: Path) -> str:
    return str(tmp_path / "contracts.json")


def test_deploy_and_execute_round_trip(storage_path: str) -> None:
    vm = SmartContractVM(ledger=None, storage_file=storage_path)
    addr = vm.deploy_contract(
        owner="USER_TEST", code=COUNTER_CODE, initial_storage={"count": 0}
    )
    assert isinstance(addr, str) and addr

    # Increment by 5 -> 5
    out = vm.execute(addr, "increment", [5])
    assert out == 5

    # Read it back
    assert vm.execute(addr, "get_count", []) == 5


def test_state_persists_across_vm_reload(storage_path: str) -> None:
    vm = SmartContractVM(ledger=None, storage_file=storage_path)
    addr = vm.deploy_contract(
        owner="USER_TEST", code=COUNTER_CODE, initial_storage={"count": 0}
    )
    assert vm.execute(addr, "increment", [7]) == 7

    # Reload from disk
    vm2 = SmartContractVM(ledger=None, storage_file=storage_path)
    assert vm2.execute(addr, "get_count", []) == 7


def test_increment_chained_calls_accumulate(storage_path: str) -> None:
    vm = SmartContractVM(ledger=None, storage_file=storage_path)
    addr = vm.deploy_contract(
        owner="USER_TEST", code=COUNTER_CODE, initial_storage={"count": 0}
    )
    for _ in range(3):
        vm.execute(addr, "increment", [2])
    assert vm.execute(addr, "get_count", []) == 6
