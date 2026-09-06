"""macOS + Apple Silicon. 밤에 켜 두는 맥이 여기다.

리눅스 쪽과 규칙이 하나 다르고, 그 하나가 이 파일의 대부분을 설명한다.

## 문지기에서 GPU 검사가 빠진다

리눅스에서는 `nvidia-smi`가 GPU에 붙은 프로세스를 하나씩 말해 주고, 큐는 그것이 비어야
다음 것을 꺼낸다. 맥에는 그 질문에 답하는 것이 없다. Metal은 어느 프로세스가 GPU를 쥐고
있는지를 밖으로 내주지 않고, `ioreg`가 주는 것은 기계 전체의 사용률 하나뿐이다.

그리고 있었어도 그대로 쓰면 안 된다. Spark는 GPU가 비어 있는 것이 기본인 기계지만, 이
맥은 **앱 자신이 늘 GPU를 쓰는** 기계다 — ollama, Whisper, Vision, 창을 그리는 일까지.
"GPU가 비어야 시작한다"를 옮겨 오면 큐는 영영 시작하지 않는다.

그래서 이 기계의 문은 `train-*` tmux 세션 하나만 본다. 겹치면 안 되는 것은 **학습 둘**
이고, 학습을 띄우는 문은 큐 하나뿐이므로 그 검사로 충분하다. 앱이 옆에서 GPU를 쓰는 것은
사고가 아니라 이 기계의 평상시다.

## 잠들면 학습이 끊긴다

Spark는 잠들지 않지만 맥은 잠든다. `pmset`을 보면 AC 전원에서도 `sleep 1`이다. 그래서
잡을 `caffeinate`로 감싼다 — 잡이 도는 동안만 붙잡고, 끝나면 놓는다. 큐를 켜 두었다는
이유로 기계가 영영 안 자면 안 되므로 데몬이 아니라 **잡이** 붙잡는 것이 요점이다.

화면은 재우게 둔다(`-d`를 쓰지 않는다). 밤새 밝은 화면을 켜 둘 이유가 없다.

한 가지 사람이 알아야 할 것: **뚜껑을 닫으면 `caffeinate`로도 못 막는다.** 외장 디스플레이
없이 덮으면 맥은 잔다. 밤새 돌리려면 뚜껑은 열어 두어야 한다.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import shutil
import subprocess
import time
from functools import lru_cache

KIND = "darwin"

#: 어느 프로세스가 GPU를 쥐고 있는지 볼 방법이 없다. 문지기에서 그 검사가 빠진다.
WATCHES_GPU_PROCESSES = False

_IOREG_UTILIZATION = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_IOREG_GPU_MEMORY = re.compile(r'"In use system memory"\s*=\s*(\d+)')
_IOREG_CORES = re.compile(r'"gpu-core-count"\s*=\s*(\d+)')
_VM_STAT_LINE = re.compile(r"^(.+?):\s+(\d+)\.?$")


def gpu_processes() -> list[dict] | None:
    """이 기계에서는 답할 수 없는 질문.

    `WATCHES_GPU_PROCESSES`가 거짓이라 문지기도 API도 이것을 부르지 않는다. 그래도
    빈 목록이 아니라 `None`을 돌려주는 이유는, 누가 나중에 이 검사를 되살렸을 때
    "확인했고 비어 있다"로 읽히면 안 되기 때문이다. 못 본 것은 못 본 것이다.
    """
    return None


def child_pids(seed: list[str]) -> set[str] | None:
    """씨앗 PID들과 그 모든 자손.

    맥에는 `/proc`이 없어 부모→자식 표를 `ps`에서 한 번에 받아 뒤집는다. PID마다
    `pgrep -P`를 부르면 프로세스가 많은 기계에서 호출이 수십 번이 된다.
    """
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    children: dict[str, list[str]] = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(parts[1], []).append(parts[0])
    pending = list(seed)
    found: set[str] = set()
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        pending.extend(children.get(pid, ()))
    return found


@lru_cache(maxsize=1)
def _gpu_name() -> str:
    """`Apple M2 Max · 38코어`. 기계가 도는 동안 바뀌지 않으므로 한 번만 읽는다."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=10,
        )
        name = out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        name = ""
    cores = _ioreg().get("cores")
    if name and cores:
        return f"{name} · {cores}코어"
    return name or "Apple Silicon"


def _ioreg() -> dict:
    """`ioreg` 한 번으로 사용률·GPU가 쥔 메모리·코어 수를 함께 읽는다. 27ms쯤 걸린다.

    상태는 3초마다 물어 오므로 이 값이 싸야 한다. `system_profiler`(0.3초)나
    `top -l 2`(1.5초, 게다가 스스로 CPU를 먹는다)는 이 자리에 둘 수 없다.
    """
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    found: dict[str, int] = {}
    if match := _IOREG_UTILIZATION.search(out.stdout):
        found["utilization"] = int(match.group(1))
    if match := _IOREG_GPU_MEMORY.search(out.stdout):
        found["gpu_memory_bytes"] = int(match.group(1))
    if match := _IOREG_CORES.search(out.stdout):
        found["cores"] = int(match.group(1))
    return found


class _CPULoad(ctypes.Structure):
    _fields_ = [("cpu_ticks", ctypes.c_uint * 4)]  # user, system, idle, nice


_HOST_CPU_LOAD_INFO = 3


@lru_cache(maxsize=1)
def _libc():
    library = ctypes.util.find_library("c")
    if library is None:
        return None
    handle = ctypes.CDLL(library, use_errno=True)
    handle.mach_host_self.restype = ctypes.c_uint
    return handle


def cpu_percent(interval: float = 0.25) -> float | None:
    """0…100. 리눅스와 같은 방식 — 누적 틱을 두 번 재 그 사이 바쁜 시간의 비율.

    `top -l 2`가 같은 답을 주지만 1.5초가 걸리고 그동안 CPU를 1.3초어치 먹는다. 사용률을
    재는 도구가 사용률을 만들면 답이 답이 아니게 된다. 커널이 들고 있는 누적값을 그대로
    두 번 읽는 쪽이 싸고 정확하다.
    """
    libc = _libc()
    if libc is None:
        return None

    def sample() -> tuple[int, int] | None:
        info = _CPULoad()
        count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint))
        if libc.host_statistics(
            libc.mach_host_self(), _HOST_CPU_LOAD_INFO, ctypes.byref(info), ctypes.byref(count)
        ) != 0:
            return None
        user, system, idle, nice = info.cpu_ticks
        return user + system + idle + nice, idle

    first = sample()
    if first is None:
        return None
    time.sleep(interval)
    second = sample()
    if second is None:
        return None
    total_delta = second[0] - first[0]
    idle_delta = second[1] - first[1]
    if total_delta <= 0:
        return None
    return round(100.0 * (total_delta - idle_delta) / total_delta, 1)


def memory() -> dict:
    """통합메모리의 전체·사용. 맥도 CPU와 GPU가 이 메모리를 나눠 쓴다.

    "쓸 수 있는 메모리"를 free + inactive + speculative로 잡는다. 리눅스의 `MemAvailable`이
    하는 것과 같은 어림이다 — inactive는 커널이 곧바로 회수할 수 있는 페이지이므로 이것을
    사용 중으로 세면 맥은 늘 가득 차 보인다.
    """
    try:
        page = int(subprocess.run(
            ["sysctl", "-n", "hw.pagesize"], capture_output=True, text=True, timeout=10,
        ).stdout.strip())
        total = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10,
        ).stdout.strip())
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    if out.returncode != 0:
        return {}
    pages: dict[str, int] = {}
    for line in out.stdout.splitlines():
        if match := _VM_STAT_LINE.match(line.strip()):
            pages[match.group(1).strip()] = int(match.group(2))
    available = sum(
        pages.get(key, 0)
        for key in ("Pages free", "Pages inactive", "Pages speculative")
    ) * page
    if total <= 0:
        return {}
    return {"total_bytes": total, "used_bytes": max(0, total - available)}


def power() -> dict:
    """AC인가 배터리인가. 이 기계에만 있는 칸이다.

    막지는 않는다 — 무엇을 언제 걸지는 사람이 정하기로 했다. 다만 배터리로 여섯 시간짜리
    학습을 거는 것은 거의 늘 실수이므로, 화면이 그것을 말할 수 있게 값은 내놓는다.
    """
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    text = out.stdout
    source = "ac" if "AC Power" in text else ("battery" if "Battery Power" in text else "")
    found: dict[str, object] = {}
    if source:
        found["source"] = source
    if match := re.search(r"(\d+)%", text):
        found["battery_percent"] = int(match.group(1))
    return found


def machine() -> dict:
    """기계 상태 가운데 이 플랫폼만 아는 것.

    온도와 전력은 `powermetrics`에 있고 그것은 sudo를 요구한다. 큐를 sudo로 돌릴 이유가
    없으므로 두 칸은 비워 둔다 — 화면은 없는 값을 이미 견딘다.
    """
    stats = _ioreg()
    info: dict[str, object] = {
        "gpu": {
            "name": _gpu_name(),
            # 통합메모리라 GPU 전용 메모리라는 것이 없다. GB10에서 nvidia-smi가 N/A로
            # 답하는 자리와 같다 — 아래 `memory`가 이 기계가 얼마나 찼는지를 말한다.
            "memory_used_mib": None,
            "memory_total_mib": None,
            "temperature_c": None,
            "power_w": None,
            "utilization_percent": stats.get("utilization"),
        },
        "cpu_percent": cpu_percent(),
        "memory": memory(),
    }
    if supply := power():
        info["power"] = supply
    return info


def wrap(command: str) -> str:
    """잡이 도는 동안 기계를 깨워 둔다.

    `-i` 유휴 잠들기, `-m` 디스크 잠들기, `-s` 시스템 잠들기를 막는다. `-d`(화면)는 일부러
    빼 두었다 — 밤새 화면을 켜 둘 이유가 없다. 잡이 끝나면 `caffeinate`도 함께 죽으므로
    빈 큐가 기계를 깨워 두는 일은 없다.
    """
    if shutil.which("caffeinate") is None:
        return command
    return f"caffeinate -i -m -s {command}"


def timeout_prefix(seconds: int) -> str | None:
    """곁다리를 시한부로 만드는 앞머리.

    맥에는 coreutils의 `timeout`이 없다. `brew install coreutils`로 깔면 `gtimeout`이
    생긴다. 없으면 `None`이고, 부르는 쪽은 시한을 강제할 수 없는 곁다리를 **띄우지 않는다**
    — 시한이 이 레인의 전부이므로, 못 지키면 시작하지 않는 것이 맞다.
    """
    binary = shutil.which("gtimeout") or shutil.which("timeout")
    if binary is None:
        return None
    return f"{binary} --signal=INT {seconds}"
