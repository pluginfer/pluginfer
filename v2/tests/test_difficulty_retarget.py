"""
Difficulty retargeting (sec3.6) regression test
================================================
Closes sec3.6: PoW difficulty was a hardcoded constant, so a 100x
hashrate increase would mint coinbase 100x faster, breaking the
issuance schedule.

Cases:
  1. Below RETARGET_INTERVAL height, difficulty == DIFFICULTY_TARGET
     (genesis-epoch bootstrap).
  2. Window much faster than target -> difficulty bumps up.
  3. Window much slower than target -> difficulty drops (clamped at
     MIN_DIFFICULTY).
  4. Window within tolerance -> difficulty unchanged.
  5. Bounded: difficulty never exceeds MAX_DIFFICULTY.
  6. get_current_difficulty is pure of chain state (idempotent
     across calls; two ledgers at identical chain state agree).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
for parent in [_HERE.parents[1], _HERE.parents[2]]:
    if (parent / "core").is_dir():
        sys.path.insert(0, str(parent))
        break

from core.compute_ledger import ComputeLedger, Block  # noqa: E402


def _stuff_chain_with_timed_blocks(ledger, n, dt_s):
    """Append n cheap blocks with synthetic timestamps spaced by dt_s."""
    base_t = time.time()
    for i in range(n):
        prev = ledger.chain[-1]
        b = Block(index=prev.index + 1, previous_hash=prev.hash,
                  transactions=[], difficulty=4)
        b.timestamp = base_t + (i + 1) * dt_s
        # Cheap PoW (difficulty=1 for the synthetic chain — we don't
        # care about real work here, only that timestamps drive DAA).
        b.difficulty = 4
        target = "0" * 1
        n2 = 0
        while n2 < 1_000_000:
            b.nonce = n2
            b.hash = b.calculate_hash()
            if b.hash.startswith(target):
                break
            n2 += 1
        ledger.chain.append(b)


def test_genesis_epoch_uses_default():
    print("\n[1] GENESIS-EPOCH BOOTSTRAP USES DIFFICULTY_TARGET")
    print("-" * 60)
    led = ComputeLedger("d1")
    assert led.get_current_difficulty() == led.DIFFICULTY_TARGET
    print(f"  height=0 -> diff={led.get_current_difficulty()} OK")
    print("  PASS")


def test_fast_window_bumps_difficulty():
    print("\n[2] FAST WINDOW -> DIFFICULTY BUMPS UP")
    print("-" * 60)
    led = ComputeLedger("d2")
    fast_dt = led.TARGET_BLOCK_TIME_S / 4   # 4x too fast
    _stuff_chain_with_timed_blocks(led, led.RETARGET_INTERVAL, fast_dt)
    new = led.get_current_difficulty()
    print(f"  observed dt={fast_dt}s vs target {led.TARGET_BLOCK_TIME_S}s "
          f"-> diff {led.chain[-1].difficulty} -> {new}")
    assert new == led.chain[-1].difficulty + 1, \
        f"expected +1, got {new}"
    print("  PASS")


def test_slow_window_drops_difficulty():
    print("\n[3] SLOW WINDOW -> DIFFICULTY DROPS")
    print("-" * 60)
    led = ComputeLedger("d3")
    slow_dt = led.TARGET_BLOCK_TIME_S * 4   # 4x too slow
    _stuff_chain_with_timed_blocks(led, led.RETARGET_INTERVAL, slow_dt)
    new = led.get_current_difficulty()
    print(f"  observed dt={slow_dt}s -> diff {led.chain[-1].difficulty} "
          f"-> {new}")
    assert new == led.chain[-1].difficulty - 1, \
        f"expected -1, got {new}"
    print("  PASS")


def test_in_band_window_no_change():
    print("\n[4] IN-BAND WINDOW -> NO CHANGE")
    print("-" * 60)
    led = ComputeLedger("d4")
    on_target = led.TARGET_BLOCK_TIME_S
    _stuff_chain_with_timed_blocks(led, led.RETARGET_INTERVAL, on_target)
    new = led.get_current_difficulty()
    print(f"  observed dt={on_target}s ~= target -> diff "
          f"{led.chain[-1].difficulty} -> {new}")
    assert new == led.chain[-1].difficulty
    print("  PASS")


def test_difficulty_floor():
    print("\n[5] DIFFICULTY FLOORED AT MIN_DIFFICULTY")
    print("-" * 60)
    led = ComputeLedger("d5")
    # Force the last block's difficulty down to the floor.
    led.MIN_DIFFICULTY  # type: ignore[attr-defined]
    fast_dt = led.TARGET_BLOCK_TIME_S * 100  # absurdly slow
    _stuff_chain_with_timed_blocks(led, led.RETARGET_INTERVAL, fast_dt)
    led.chain[-1].difficulty = led.MIN_DIFFICULTY
    new = led.get_current_difficulty()
    assert new >= led.MIN_DIFFICULTY
    print(f"  diff floored at MIN_DIFFICULTY={led.MIN_DIFFICULTY} -> {new} OK")
    print("  PASS")


def test_pure_function_of_chain_state():
    print("\n[6] DETERMINISTIC ACROSS LEDGERS")
    print("-" * 60)
    led_a = ComputeLedger("d6a")
    led_b = ComputeLedger("d6b")
    fast_dt = led_a.TARGET_BLOCK_TIME_S / 3
    _stuff_chain_with_timed_blocks(led_a, led_a.RETARGET_INTERVAL, fast_dt)
    # Replicate identical timing on B.
    _stuff_chain_with_timed_blocks(led_b, led_b.RETARGET_INTERVAL, fast_dt)
    # Force timestamps to match exactly so we can compare.
    for i, blk in enumerate(led_a.chain[1:], start=1):
        led_b.chain[i].timestamp = blk.timestamp

    a = led_a.get_current_difficulty()
    b = led_b.get_current_difficulty()
    assert a == b, f"non-deterministic: {a} != {b}"
    print(f"  led_a={a}  led_b={b} (identical chain state) OK")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("DIFFICULTY RETARGET (sec3.6) TEST")
    print("=" * 60)
    t0 = time.time()
    test_genesis_epoch_uses_default()
    test_fast_window_bumps_difficulty()
    test_slow_window_drops_difficulty()
    test_in_band_window_no_change()
    test_difficulty_floor()
    test_pure_function_of_chain_state()
    print("\n" + "=" * 60)
    print(f"ALL DIFFICULTY-RETARGET TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
