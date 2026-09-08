#!/usr/bin/env python3
"""LeRobot의 policy_server를 두 가지 점에서 감싼다 — 적재 캐시와 RTC.

## 1. 같은 체크포인트를 두 번 적재하지 않는다

`SendPolicyInstructions`는 클라이언트가 붙을 때마다 `from_pretrained()`를 무조건 다시
부른다(`policy_server.py:152`). π₀.₅ 한 벌(9.35GB)에 이 기계에서 **131초**가 걸리고, 시행마다
클라이언트가 새로 붙으므로 자리 5곳 × 4회짜리 성공률 측정이면 적재에만 44분이 간다.

## 2. 새 청크를 앞 계획에 이어 붙인다 (Real-Time Chunking)

`RobotClientConfig`에는 RTC 자리가 없다. 그래서 원격 추론은 **지금 팔이 무엇을 실행 중인지
모른 채** 청크를 만들고, 클라이언트가 겹치는 구간을 `0.3*old + 0.7*new`로 섞어 갈아 끼운다.
2026-09-07 실측으로 그 교체가 **24프레임(0.80초)마다** 일어났고 팔이 그 주기로 툭툭 끊겼다.
2026-09-06에 로컬 경로에서 겪은 것과 같은 고장이다(그때는 0.73초). 로컬은 그때
`RTC_EXECUTION_HORIZON = 25`로 고쳤는데, 원격은 그 수정을 받은 적이 없다.

RTC에 필요한 두 값은 **서버가 스스로 구할 수 있다.** 클라이언트가 관측마다 `timestep`을
싣고 오므로, 서버는 자기가 마지막으로 낸 청크가 어느 timestep에서 시작했는지와 견주어
"아직 실행되지 않은 꼬리"를 그대로 잘라낼 수 있다. 프로토콜을 바꿀 필요가 없다.

체크포인트에는 `rtc_config`가 없어서(`PI05Config.rtc_config`의 기본값은 `None`) RTC 처리기
자체가 만들어지지 않는다. 여기서 만들어 넣는다. 기본 `execution_horizon`은 **10**인데, 그
값이 2026-09-06 사고의 원인이었으므로 쓰지 않는다.

고치는 자리를 site-packages가 아니라 여기로 잡은 이유: 그 파일을 고치면 `uv sync`나
lerobot 재설치가 조용히 되돌린다.
"""

from __future__ import annotations

import math
import time

from lerobot.async_inference import policy_server

#: 새 청크를 앞 계획에 붙들어 두는 구간(프레임).
#:
#: 로컬 경로가 2026-09-06에 고른 것과 같은 값이고, 이유도 같다. 큐가 버리는 앞부분(추론이
#: 걸린 프레임 수)보다 커야 실제로 실행되는 구간에 연속성 제약이 남는다. 이 기계의 원격
#: 왕복은 0.33~0.67초, 30fps로 10~20프레임이므로 25가 그 위에 선다. lerobot은 남은 꼬리가
#: 이보다 짧으면 스스로 꼬리 길이로 줄인다(`modeling_rtc.py`), 그래서 위쪽만 정하면 된다.
RTC_EXECUTION_HORIZON = 25

_original_get_policy_class = policy_server.get_policy_class
_original_get_action_chunk = policy_server.PolicyServer._get_action_chunk
_original_predict_action_chunk = policy_server.PolicyServer._predict_action_chunk
_original_reset_server = policy_server.PolicyServer._reset_server

_LOADED: dict[tuple[str, str], object] = {}


class _CachedPolicyClass:
    """`get_policy_class`가 돌려주던 클래스의 자리에 서서, 같은 경로면 든 것을 다시 준다."""

    def __init__(self, policy_class: type) -> None:
        self._policy_class = policy_class

    def from_pretrained(self, path, *args, **kwargs):
        key = (self._policy_class.__name__, str(path))
        cached = _LOADED.get(key)
        if cached is not None:
            print(f"[soarm] 이미 든 정책을 그대로 씁니다 — 적재 건너뜀: {key[1]}", flush=True)
            return cached
        _LOADED.clear()
        print(f"[soarm] 정책을 처음 듭니다 — 이 체크포인트는 이번 한 번만 오래 걸립니다: {key[1]}", flush=True)
        policy = self._policy_class.from_pretrained(path, *args, **kwargs)
        _enable_rtc(policy)
        _LOADED[key] = policy
        return policy

    def __getattr__(self, name: str):
        return getattr(self._policy_class, name)


def _enable_rtc(policy) -> None:
    """체크포인트에 없는 RTC 설정을 만들어 넣는다. 못 하면 조용히 없이 간다."""
    try:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        policy.config.rtc_config = RTCConfig(execution_horizon=RTC_EXECUTION_HORIZON)
        policy.init_rtc_processor()
    except Exception as error:  # RTC를 모르는 정책·버전이면 그냥 안 쓴다
        print(f"[soarm] RTC를 켜지 못했습니다({error}). 이어 붙이기 없이 돕니다.", flush=True)
        return
    enabled = getattr(policy, "rtc_processor", None) is not None
    print(
        f"[soarm] RTC {'켜짐' if enabled else '꺼짐'} · execution_horizon={RTC_EXECUTION_HORIZON}",
        flush=True,
    )


class _RealTimeChunking:
    """마지막으로 낸 청크를 들고 있다가, 다음 청크를 그 남은 꼬리 위에 이어 붙이게 한다."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        #: 지금 처리 중인 관측의 timestep. 클라이언트가 실어 보낸 값이다.
        self.timestep: int | None = None
        #: 마지막으로 낸 청크. **후처리 전**의 값이어야 한다 — RTC의 유도는 정책이 내놓는
        #: 공간에서 이루어지고, 역정규화를 거친 값을 넣으면 엉뚱한 곳으로 끌어당긴다.
        self.chunk = None
        #: 그 청크의 첫 액션이 놓인 timestep.
        self.chunk_timestep: int | None = None
        #: 최근 추론이 걸린 프레임 수. 큐가 새 청크의 앞부분을 그만큼 버린다.
        self.delay = 0

    def left_over(self):
        """아직 실행되지 않은 앞 계획의 꼬리. 없으면 `None`."""
        if self.chunk is None or self.chunk_timestep is None or self.timestep is None:
            return None
        consumed = self.timestep - self.chunk_timestep
        if consumed <= 0 or consumed >= self.chunk.shape[1]:
            return None
        return self.chunk[:, consumed:, :]

    def get_action_chunk(self, server, observation):
        guided = getattr(server.policy, "rtc_processor", None) is not None
        previous = self.left_over() if guided else None
        started = time.perf_counter()
        if previous is None:
            chunk = _original_get_action_chunk(server, observation)
        else:
            # 유도 구간은 꼬리보다 길 수 없고(lerobot이 스스로 줄인다), 지연은 그 안에
            # 들어와야 실제로 실행되는 부분에 제약이 남는다.
            horizon = min(RTC_EXECUTION_HORIZON, int(previous.shape[1]))
            delay = max(0, min(self.delay, horizon - 1))
            chunk = server.policy.predict_action_chunk(
                observation, inference_delay=delay, prev_chunk_left_over=previous
            )
            if chunk.ndim != 3:
                chunk = chunk.unsqueeze(0)
            chunk = chunk[:, : server.actions_per_chunk, :]
        self._remember(server, chunk, time.perf_counter() - started)
        # 이어 붙였는지 아닌지는 로그로 남는다. 이 줄이 없으면 RTC가 켜졌다는 것과 실제로
        # 걸렸다는 것을 구별할 방법이 없다 — 꼬리가 없으면 켜져 있어도 걸리지 않는다.
        if previous is None:
            print("[soarm] 이어 붙일 앞 계획이 없어 새로 만듭니다", flush=True)
        else:
            print(
                f"[soarm] 앞 계획 {int(previous.shape[1])}프레임 위에 이어 붙였습니다 "
                f"(버려질 앞부분 {self.delay}프레임)",
                flush=True,
            )
        return chunk

    def _remember(self, server, chunk, seconds: float) -> None:
        fps = getattr(server.config, "fps", 30) or 30
        self.delay = max(0, math.ceil(seconds * fps))
        self.chunk = chunk.detach().clone()
        self.chunk_timestep = self.timestep


_RTC = _RealTimeChunking()


def _cached_get_policy_class(name: str) -> _CachedPolicyClass:
    return _CachedPolicyClass(_original_get_policy_class(name))


def _patched_predict_action_chunk(self, observation_t):
    # 이 관측이 놓인 자리를 기억해 둔다. 아래 `_get_action_chunk`가 앞 계획과 견주는 기준이다.
    _RTC.timestep = observation_t.get_timestep()
    return _original_predict_action_chunk(self, observation_t)


def _patched_get_action_chunk(self, observation):
    return _RTC.get_action_chunk(self, observation)


def _patched_reset_server(self):
    # 새 클라이언트가 붙었다. 앞 계획은 지난 시행의 것이므로 이어 붙일 대상이 아니다.
    _RTC.reset()
    return _original_reset_server(self)


def main() -> None:
    policy_server.get_policy_class = _cached_get_policy_class
    policy_server.PolicyServer._get_action_chunk = _patched_get_action_chunk
    policy_server.PolicyServer._predict_action_chunk = _patched_predict_action_chunk
    policy_server.PolicyServer._reset_server = _patched_reset_server
    # `serve`는 draccus로 감싸여 있어 `--host/--port/--fps`를 sys.argv에서 읽는다.
    policy_server.serve()


if __name__ == "__main__":
    main()
