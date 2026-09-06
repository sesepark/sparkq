"""기계에 묶인 것들만 모아 둔 자리.

큐의 규칙 — 슬롯은 하나, 비선점, 상태는 디스크에, 실패해도 다음으로 — 은 기계를 가리지
않는다. 기계를 가리는 것은 그 규칙을 **확인하는 방법**뿐이다. GPU가 비었는지 무엇으로
보는가, CPU 사용률을 어디서 읽는가, 잠들지 않게 하려면 무엇으로 감싸는가.

그래서 그 다섯 가지만 여기로 뺐다. `sparkq.py`는 이 이름들만 부르고, 어느 기계에서 도는지
알지 못한다.

## 왜 "GPU를 볼 수 있는가"가 이 층의 값인가

리눅스에서 큐가 다음 것을 꺼내는 조건은 `train-*` 세션이 없고 **GPU 컴퓨트 프로세스도
하나도 없을 때**다. 두 번째 검사가 있는 이유는 사람이 터미널에서 직접 띄운 학습 위에
올라타지 않기 위해서다.

맥에는 그 검사가 **존재할 수 없다.** `nvidia-smi --query-compute-apps`에 해당하는 것이
없어서, 어느 프로세스가 GPU를 쥐고 있는지 볼 방법이 없다. 그리고 맥에서는 그 검사가
있어도 곤란하다 — 이 기계에서는 앱 자신이 늘 GPU를 쓴다(ollama, Whisper, Vision). "GPU가
비어야 시작한다"를 그대로 옮기면 큐는 영영 시작하지 않는다.

그래서 `WATCHES_GPU_PROCESSES`가 이 층의 값으로 나와 있다. 거짓이면 문지기는 세션만 보고,
**API도 `gpu_apps`를 아예 싣지 않는다.** 빈 목록으로 실으면 "확인했고 비어 있다"는 뜻이
되는데, 실제로는 확인할 수 없었던 것이다. 화면이 그 둘을 구별할 수 있어야 한다.
"""

from __future__ import annotations

import sys

if sys.platform == "darwin":
    from . import darwin as _impl
else:
    from . import linux as _impl

#: 이 기계가 어느 쪽인가. API의 `capabilities.platform`으로 나간다.
KIND: str = _impl.KIND

#: GPU를 쥔 프로세스를 하나씩 볼 수 있는가. 거짓이면 문지기에서 그 검사가 빠진다.
WATCHES_GPU_PROCESSES: bool = _impl.WATCHES_GPU_PROCESSES

gpu_processes = _impl.gpu_processes
child_pids = _impl.child_pids
machine = _impl.machine
wrap = _impl.wrap
timeout_prefix = _impl.timeout_prefix

__all__ = [
    "KIND",
    "WATCHES_GPU_PROCESSES",
    "gpu_processes",
    "child_pids",
    "machine",
    "wrap",
    "timeout_prefix",
]
