#!/usr/bin/env python3
"""LeRobot training with safe checkpoint-and-exit support for policy inference."""

from __future__ import annotations

import signal
import sys

from lerobot.scripts import lerobot_train


CHECKPOINTED_FOR_PREEMPTION = 75
_requested = False


class _CheckpointedForPreemption(Exception):
    pass


def _request(_signum: int, _frame: object) -> None:
    global _requested
    _requested = True
    print("\n[sparkq] 정책 추론 요청: 현재 스텝을 마친 뒤 체크포인트를 저장합니다.", flush=True)


_normal_should_save = lerobot_train.should_save_checkpoint
_normal_update_last = lerobot_train.update_last_checkpoint


def _should_save(*args: object, **kwargs: object) -> bool:
    return _requested or _normal_should_save(*args, **kwargs)


def _update_last(*args: object, **kwargs: object) -> object:
    result = _normal_update_last(*args, **kwargs)
    if _requested:
        print("[sparkq] 선점 체크포인트가 완성됐습니다. GPU 메모리를 반환합니다.", flush=True)
        raise _CheckpointedForPreemption
    return result


def main() -> int:
    signal.signal(signal.SIGUSR1, _request)
    lerobot_train.should_save_checkpoint = _should_save
    lerobot_train.update_last_checkpoint = _update_last
    try:
        lerobot_train.main()
    except _CheckpointedForPreemption:
        return CHECKPOINTED_FOR_PREEMPTION
    return 0


if __name__ == "__main__":
    sys.exit(main())
