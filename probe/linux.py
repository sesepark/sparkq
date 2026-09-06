"""리눅스 + NVIDIA. DGX Spark가 여기다.

여기 있는 코드는 전부 `sparkq.py`에 있던 것을 그대로 옮긴 것이다. 옮기면서 고친 것은
없다 — 이 경로는 밤새 도는 학습들이 이미 지나간 길이라, 갈라내는 김에 손보는 것은
갈라내기가 잘못됐는지 학습이 잘못됐는지 알 수 없게 만든다.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

KIND = "linux"

#: nvidia-smi가 GPU에 붙은 프로세스를 하나씩 말해 준다. 그래서 문지기가 그것을 본다.
WATCHES_GPU_PROCESSES = True


def gpu_processes() -> list[dict] | None:
    """GPU에 붙어 있는 컴퓨트 프로세스. 화면(Xorg·gnome-shell)은 여기에 안 잡힌다.

    **못 읽은 것과 없는 것을 가른다.** nvidia-smi가 잠깐 실패해 빈 목록을 돌려주면,
    그것을 `GPU가 비었다`로 읽는 순간 큐는 남이 돌리는 학습 위에 하나 더 띄운다 —
    통합메모리라 그때 둘 다 스왑으로 느려진다. 그래서 실패는 `None`이고, 그것을 받은
    쪽은 다음 것을 꺼내지 않고 기다린다. 비어 있는 것(`[]`)만이 시작해도 좋다는 뜻이다.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) >= 3:
            row = {"pid": parts[0], "name": parts[1], "memory_mib": parts[2]}
            row.update(_describe_process(parts[0]))
            rows.append(row)
    return rows


def _describe_process(pid: str) -> dict:
    """큐 밖의 프로세스가 **무엇**인지. nvidia-smi가 주는 것은 실행 파일 경로뿐인데, 이 기계에서
    그것은 거의 늘 `python3`라 아무것도 말해 주지 않는다. 컨테이너 안의 프로세스도 호스트의
    /proc에서 보이므로 인자와 시작 시각은 거기서 읽는다. 못 읽으면 빈 dict — 막지 않는다.

    `label`은 사람이 한눈에 알아볼 한 줄이다: 스크립트 이름과 `--task=` 같은 결정적인 인자.
    """
    info: dict[str, object] = {}
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return info
    args = [piece.decode("utf-8", errors="replace") for piece in raw.split(b"\0") if piece]
    if not args:
        return info
    info["cmdline"] = " ".join(args)[:400]
    script = next((piece for piece in args if piece.endswith(".py")), None)
    label = Path(script).name if script else Path(args[0]).name
    decisive = [piece for piece in args[1:] if piece.startswith(("--task", "--dataset.repo_id", "--policy."))]
    if decisive:
        label += " " + " ".join(decisive)[:120]
    info["label"] = label
    try:
        out = subprocess.run(["ps", "-o", "etimes=", "-p", pid], capture_output=True, text=True, timeout=5)
        info["elapsed_seconds"] = int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return info


def child_pids(seed: list[str]) -> set[str] | None:
    """씨앗 PID들과 그 모든 자손. 커널이 자식 목록을 직접 준다."""
    pending = list(seed)
    found: set[str] = set()
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8")
        except OSError:
            continue
        pending.extend(child for child in children.split() if child.isdigit())
    return found


def cpu_percent(interval: float = 0.25) -> float | None:
    """0…100. /proc/stat을 두 번 재 그 사이 바쁜 시간의 비율을 구한다.

    한 번만 읽으면 부팅 이후 누적값이라 순간 사용률이 아니다. 두 시각의 차를 봐야 한다.
    """
    def sample() -> tuple[int, int] | None:
        try:
            with open("/proc/stat", encoding="utf-8") as handle:
                parts = handle.readline().split()
        except OSError:
            return None
        if len(parts) < 5 or parts[0] != "cpu":
            return None
        values = [int(v) for v in parts[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
        return sum(values), idle

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
    """통합메모리의 전체·사용. GB10은 CPU와 GPU가 이 메모리를 나눠 쓰므로, RAM 사용률이
    곧 이 기계가 얼마나 찼는지다 — nvidia-smi의 GPU 전용 메모리는 여기서 N/A로 나온다."""
    total = available = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return {}
    if total is None or available is None:
        return {}
    return {"total_bytes": total, "used_bytes": total - available}


def machine() -> dict:
    """기계 상태 가운데 이 플랫폼만 아는 것. 호스트 이름·디스크·큐 길이는 부르는 쪽이 얹는다."""
    info: dict[str, object] = {}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, used, total, temperature, power, util = [
                piece.strip() for piece in out.stdout.strip().splitlines()[0].split(",")
            ]

            def number(text, cast):
                try:
                    return cast(float(text))
                except ValueError:
                    return None

            info["gpu"] = {
                "name": name,
                "memory_used_mib": number(used, int),
                "memory_total_mib": number(total, int),
                "temperature_c": number(temperature, int),
                "power_w": number(power, float),
                "utilization_percent": number(util, int),
            }
    except (OSError, subprocess.SubprocessError) as error:
        info["gpu_error"] = str(error)
    info["cpu_percent"] = cpu_percent()
    info["memory"] = memory()
    return info


def wrap(command: str) -> str:
    """이 기계는 잠들지 않는다. 감쌀 것이 없다."""
    return command


def timeout_prefix(seconds: int) -> str | None:
    """곁다리를 시한부로 만드는 앞머리. coreutils의 `timeout`이 늘 있다."""
    return f"timeout --signal=INT {seconds}"
