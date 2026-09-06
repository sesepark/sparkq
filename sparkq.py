#!/usr/bin/env python3
"""sparkq — DGX Spark의 GPU 한 대 앞에 줄을 세우는 작은 실행기.

이 기계의 희소 자원은 GB10 하나다. 그래서 슬롯은 하나이고, 큐는 한 번에 하나만 돌린다.
통합메모리라 두 학습이 겹치면 OOM으로 깨지는 대신 **둘 다 스왑으로 느려지는** 방식으로
망가지는데, 그쪽이 훨씬 알아채기 어렵다. 겹치지 않게 하는 것이 이 프로그램의 전부다.

## 왜 데몬이 파일을 읽는 구조인가

큐를 메모리에 들고 있으면 데몬이 죽는 순간 밤새 세워 둔 줄이 사라진다. 상태를 전부 디스크에
두면 데몬은 그 파일을 읽어 실행하는 얇은 층이 되고, 재시작·재부팅·손으로 고치기를 전부
견딘다. 실제로 `sparkq`를 `systemctl --user restart` 해도 도는 학습은 tmux 안에서 그대로
돌고, 올라온 데몬이 그것을 다시 찾아 붙는다.

## 왜 비선점인가

학습은 몇 시간짜리이고 중간에 뺏으면 처음부터 다시 해야 한다. 그래서 한 번 시작한 작업은
끝나거나 사람이 세울 때까지 둔다. 대신 **아직 시작하지 않은** 줄은 얼마든지 다시 세울 수
있게 한다(`top`, `rm`).

## 큐 밖에서 손으로 띄운 작업

이 기계에서는 사람이 터미널에서 직접 학습을 띄우기도 한다. 큐가 "내 앞 작업이 끝났나"만
보면 그런 작업 위에 올라타 버린다. 그래서 다음 것을 꺼내는 조건은 **GPU에 컴퓨트 프로세스가
하나도 없고 `train-*` tmux 세션도 없을 때**다. 누가 띄웠든 상관없다. CUDA 초기화 전에는
학습 세션이 이미 생겼어도 아직 GPU 프로세스로 보이지 않으므로 두 신호가 모두 필요하다.

상태는 전부 `~/.sparkq/` 아래에 있고, 조작은 `127.0.0.1`에만 열리는 HTTP로 한다. 포트를
LAN에 열지 않는 것은 이 파이프라인의 다른 서버들과 같은 규칙이고, 신뢰 경계는 SSH 터널이다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template

# `~/.local/bin/sparkq` 심볼릭 링크로 불릴 때도 옆의 `probe/`를 찾을 수 있게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe  # noqa: E402 — 위의 경로 조정 뒤에 와야 한다

HOME = Path.home()
ROOT = Path(os.environ.get("SPARKQ_ROOT", HOME / ".sparkq"))
QUEUE_DIR = ROOT / "queue"
RUNS_DIR = ROOT / "runs"
PAUSED_FILE = ROOT / "paused"
KINDS_DIR = Path(os.environ.get("SPARKQ_KINDS", Path(__file__).resolve().parent / "kinds"))
DATASET_ROOT = Path(os.environ.get("SPARKQ_DATASETS", HOME / "data" / "soarm"))
#: 학습이 결과를 쌓는 곳. `lerobot-resume`이 이어붙일 실행을 여기서 찾는다.
OUTPUT_ROOT = Path(os.environ.get("SPARKQ_OUTPUTS", HOME / "outputs"))
PORT = int(os.environ.get("SPARKQ_PORT", "8092"))

#: 이 기계가 GPU를 쥔 프로세스를 하나씩 볼 수 있는가.
#:
#: 문지기가 `train-*` 세션 말고 GPU까지 보는지를 가른다. `probe`의 값을 여기 한 번
#: 옮겨 두는 이유는 시험 때문이다 — 어느 기계에서 돌리든 두 경로를 모두 지나야 하는데,
#: 시험이 `probe`를 통째로 바꿔 끼우는 것보다 이 이름 하나를 바꾸는 편이 낫다.
WATCHES_GPU_PROCESSES = probe.WATCHES_GPU_PROCESSES

#: 작업 종류·데이터셋·실행 이름이 모두 이 규칙을 지난다. 이 값들은 곧 셸 명령의 일부가
#: 되므로, 검사 지점을 여러 곳에 나눠 두지 않고 여기 하나만 둔다.
NAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")

#: 기본 우선순위. `top`은 지금 줄에서 가장 작은 값보다 하나 더 작은 값을 준다.
DEFAULT_PRIORITY = 5000

# 곁다리는 짧고 사람이 보는 작업만 받는다. 종류 파일이 더 큰 값을 주장해도 읽지 않는다.
MAX_SIDE_SECONDS = 3600
CURRENT_SIDE_FILE = ROOT / "current_side"

# HTTP 요청은 ThreadingHTTPServer의 스레드에서, tick은 데몬의 주 스레드에서 돈다. 큐 파일을
# 실행 자리로 옮기는 동안 DELETE가 끼어들어 취소한 작업을 다시 띄우지 못하게 둘을 직렬화한다.
QUEUE_LOCK = threading.Lock()


class Invalid(ValueError):
    """사람이 고칠 수 있는 잘못. HTTP에서는 400으로 나간다."""


class Missing(LookupError):
    """찾는 것이 없다. HTTP에서는 404로 나간다."""


class Conflict(RuntimeError):
    """이미 차지한 단일 슬롯. HTTP에서는 409로 나간다."""


# ---------------------------------------------------------------- 작업 종류

def load_kinds() -> dict[str, dict]:
    """`kinds/*.json`에 적힌 작업 종류들.

    종류를 코드가 아니라 파일로 두는 이유는 늘리기 위해서다. 새 실험을 큐에 걸 수 있게
    하는 데 이 파일도 앱도 고칠 필요가 없고, JSON 하나를 더 놓으면 된다. 대신 그 JSON이
    셸 명령을 만들므로, 값은 아래 `validate`가 형식별로 전부 검사한다.
    """
    kinds: dict[str, dict] = {}
    for path in sorted(KINDS_DIR.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"[sparkq] 종류 파일을 읽지 못했습니다 {path}: {error}", file=sys.stderr)
            continue
        if not isinstance(spec, dict) or not NAME.match(str(spec.get("kind", ""))):
            print(f"[sparkq] 종류 파일을 읽지 않습니다 {path}: kind가 올바르지 않습니다", file=sys.stderr)
            continue
        spec = dict(spec)
        lane = spec.get("lane", "queue")
        if lane not in {"queue", "side"}:
            print(f"[sparkq] 종류 파일을 읽지 않습니다 {path}: lane은 queue 또는 side여야 합니다", file=sys.stderr)
            continue
        spec["lane"] = lane
        if lane == "side":
            limit = spec.get("limit_seconds")
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SIDE_SECONDS:
                print(
                    f"[sparkq] 종류 파일을 읽지 않습니다 {path}: side의 limit_seconds는 "
                    f"1..{MAX_SIDE_SECONDS} 정수여야 합니다",
                    file=sys.stderr,
                )
                continue
        kinds[spec["kind"]] = spec
    return kinds


def validate(spec: dict, params: dict) -> dict:
    """사람이 고른 값을 형식별로 검사하고, 프리셋이 딸려 오면 함께 펼친다.

    `text` 형식이 없는 것은 일부러다. 자유 문자열이 셸 명령에 들어가는 순간 이 큐는 원격
    셸이 된다. 늘리고 싶으면 종류 파일을 하나 더 만드는 쪽이 맞다.
    """
    values: dict[str, object] = {}
    for field in spec.get("fields", []):
        name = field["name"]
        label = field.get("label", name)
        raw = params.get(name, field.get("default"))
        form = field.get("type", "name")
        if raw is None or raw == "":
            raise Invalid(f"{label}을(를) 정해야 합니다")
        if form == "enum":
            if raw not in field.get("values", []):
                raise Invalid(f"{label}: 고를 수 없는 값입니다 ({raw})")
            values[name] = raw
            for key, value in (field.get("presets") or {}).get(raw, {}).items():
                values[key] = value
        elif form == "int":
            try:
                number = int(raw)
            except (TypeError, ValueError):
                raise Invalid(f"{label}: 숫자여야 합니다") from None
            low, high = int(field.get("min", 1)), int(field.get("max", 10_000_000))
            if not low <= number <= high:
                raise Invalid(f"{label}: {low}에서 {high} 사이여야 합니다")
            values[name] = number
        elif form == "name":
            if not NAME.match(str(raw)):
                raise Invalid(f"{label}: 쓸 수 없는 이름입니다 ({raw})")
            # 목록에서 고르는 칸은 **걸 때** 그 목록에 있는지 본다. 없는 이름으로 걸면
            # 작업은 새벽에 시작해 몇 초 만에 죽고 큐는 다음으로 넘어간다 — 아침에 남는
            # 것은 실패 한 줄과 날아간 밤 하나다. 여기서 400으로 막으면 사람이 지금 고친다.
            if source := field.get("source"):
                known = source_names(str(source))
                if known is not None and str(raw) not in known:
                    raise Invalid(f"{label}: 이 기계에 없습니다 ({raw})")
            values[name] = str(raw)
        else:
            raise Invalid(f"{label}: 알 수 없는 항목 형식({form})")
    return values


def expand(spec: dict, values: dict, job_id: str) -> dict:
    """치환에 쓸 값 전부. `${이름}` 자리에 들어간다.

    `str.format`이 아니라 `string.Template`을 쓰는 이유가 있다. 만들어 내는 것이 셸
    스크립트라 `{}`가 그대로 들어가는 자리가 있고(JSON을 적는 줄이 그렇다), `format`은
    그것을 자기 문법으로 읽고 깨진다. `safe_substitute`는 우리가 모르는 `$?`나
    `$LD_PRELOAD`를 건드리지 않고 지나간다.
    """
    now = datetime.now()
    base: dict[str, object] = dict(values)
    base.update({
        "id": job_id,
        "home": str(HOME),
        "stamp": now.strftime("%Y%m%d_%H%M%S"),
        "date": now.strftime("%Y-%m-%d"),
        # 작업 id의 끝 조각(무작위 hex). run 이름을 확실히 갈라 준다 — 시각만으로는 같은
        # 초에 두 번 걸면 겹치고, 그러면 둘째 학습이 output_dir 충돌로 죽는다.
        "token": job_id.rsplit("-", 1)[-1],
        "run_dir": str(RUNS_DIR / job_id),
        "log": str(RUNS_DIR / job_id / "run.log"),
        "dataset_root": str(DATASET_ROOT),
        "output_root": str(OUTPUT_ROOT),
    })
    # 긴 이름을 그대로 실행 이름에 넣으면 `NAME`(80자)을 넘긴다. 잘라 쓸 수 있게 짧은
    # 짝을 함께 내놓는다 — `${dataset}` 옆에 `${dataset_short}`.
    for key, value in list(base.items()):
        if isinstance(value, str):
            base[f"{key}_short"] = value[:56]
    for entry in spec.get("derived", []):
        base[entry["name"]] = Template(entry["template"]).safe_substitute(base)
    return base


# ---------------------------------------------------------------- 큐 파일

def new_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + os.urandom(2).hex()


def queue_path(priority: int, created_ns: int, job_id: str) -> Path:
    """파일 이름 하나가 곧 순서다.

    이름순으로 정렬하면 우선순위 → 들어온 순서가 된다. 순서를 따로 적은 색인 파일을 두면
    그 파일과 실제 작업 목록이 어긋나는 날이 오는데, 이름에 실어 두면 어긋날 자리가 없다.
    """
    return QUEUE_DIR / f"{priority:04d}-{created_ns:019d}-{job_id}.json"


def queued_files() -> list[Path]:
    return sorted(QUEUE_DIR.glob("*.json"))


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, payload: dict) -> None:
    """반쯤 쓰인 파일이 목록에 잡히지 않도록 옆에 쓰고 옮긴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_script_text(command: str, log: Path) -> str:
    """명령의 종료 코드를 보존하고, 0과 함께 나온 uncaught Python 예외도 실패로 바꾼다.

    Isaac Lab/Hydra 조합은 traceback을 출력한 뒤 0으로 끝나는 경로가 실제로 있다. 큐 파일은
    걸 때 명령을 확정하므로 이 검사는 종류 파일이 아니라 실행 스크립트에 두어, 이미 기다리는
    작업에도 적용한다.
    """
    quoted_log = shlex.quote(str(log))
    return (
        "#!/usr/bin/env bash\nset -o pipefail\n"
        + command
        + "\nstatus=$?\n"
        + f"if [ \"$status\" -eq 0 ] && grep -Fq -e 'Traceback (most recent call last):' "
        + f"-e 'Error executing job with overrides:' {quoted_log}; then\n"
        + "  echo '[sparkq] traceback과 함께 종료 코드 0을 받아 실패로 기록합니다.' >&2\n"
        + "  status=1\nfi\nexit \"$status\"\n"
    )


def enqueue(kind: str, params: dict, *, priority: int = DEFAULT_PRIORITY) -> dict:
    """무엇을 실행할지를 **지금** 정해서 파일로 남긴다.

    명령 문자열을 꺼낼 때가 아니라 걸 때 만드는 이유는, 대기열에 선 작업이 무엇을 실행할
    것인지를 사람이 미리 읽을 수 있어야 하기 때문이다. 자는 동안 도는 것들이므로 더욱 그렇다.
    """
    kinds = load_kinds()
    spec = kinds.get(kind)
    if spec is None:
        raise Invalid(f"알 수 없는 작업 종류입니다: {kind}")
    if spec["lane"] != "queue":
        raise Invalid(f"곁다리 종류는 /api/side로 시작해야 합니다: {kind}")
    values = validate(spec, params or {})
    job_id = new_id()
    full = expand(spec, values, job_id)
    created_ns = time.time_ns()
    job = {
        "id": job_id,
        "kind": kind,
        "title": Template(spec.get("title", kind)).safe_substitute(full),
        "params": {key: values[key] for key in (f["name"] for f in spec.get("fields", [])) if key in values},
        # 프리셋까지 펼친 값 전부. `params`는 사람이 고른 것이고 이쪽은 그것이 무엇으로
        # 풀렸는가다 — 진행률의 분모(`steps`)가 여기서만 나온다.
        "values": values,
        "command": Template(spec["run"]).safe_substitute(full),
        "session": _session_name(spec, full),
        "progress": spec.get("progress", "none"),
        "priority": priority,
        "created_at": created_ns / 1e9,
        "state": "queued",
    }
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(queue_path(priority, created_ns, job_id), job)
    return job


def _session_name(spec: dict, values: dict) -> str:
    """tmux 세션 이름. queue는 ``train-``, side는 ``side-``로 시작한다.

    이 기계에는 학습을 띄우는 문이 둘이다. 이 큐와, 팔이 붙은 HUB 콘솔의
    `POST /api/spark/train`이다. 콘솔은 학습을 띄우기 전에 원격의 tmux 세션 가운데
    `train-`으로 시작하는 것이 있는지를 보고 있으면 거절한다 — 그것이 콘솔이 아는 유일한
    "이미 도는 학습"의 표시다.

    그래서 queue가 다른 이름을 쓰면, 콘솔은 이 큐의 작업을 **보지 못하고** 그 위에 학습을
    하나 더 띄운다. GPU는 하나이고, 통합메모리라 그때 생기는 일은 OOM이 아니라 둘 다
    느려지는 것이다. 접두사를 여기서 강제하면 종류 파일을 새로 만드는 사람이 이 사정을
    몰라도 안전하다.

    반대 방향도 같은 접두사로 막는다. 콘솔이 먼저 세션을 만든 뒤 CUDA 초기화 전까지는
    GPU 컴퓨트 프로세스로 잡히지 않을 수 있으므로, 이 큐도 `train-*` 세션이 비기 전에는
    다음 것을 꺼내지 않는다. 곁다리는 그 검사에 걸리지 않도록 별개의 `side-`를 강제한다.
    """
    prefix = "side-" if spec.get("lane", "queue") == "side" else "train-"
    name = Template(spec.get("session", "sparkq-${id}")).safe_substitute(values)
    if name.startswith(prefix):
        return name
    # 잘못 쓴 side 종류가 train-으로 시작해도 train-side-...가 되지 않게 완전히 갈아 끼운다.
    if prefix == "side-" and name.startswith("train-"):
        name = name.removeprefix("train-")
    return prefix + name


def find_queued(job_id: str) -> Path | None:
    for path in queued_files():
        if path.stem.endswith("-" + job_id):
            return path
    return None


def move_to_top(job_id: str) -> dict:
    path = find_queued(job_id)
    if path is None:
        raise Missing(job_id)
    job = read_json(path) or {}
    lowest = DEFAULT_PRIORITY
    for other in queued_files():
        try:
            lowest = min(lowest, int(other.name.split("-", 1)[0]))
        except ValueError:
            pass
    priority = max(0, lowest - 1)
    job["priority"] = priority
    created_ns = int(round(float(job.get("created_at", time.time())) * 1e9))
    write_json(queue_path(priority, created_ns, job_id), job)
    path.unlink(missing_ok=True)
    return job


# ---------------------------------------------------------------- 실행 상태

def run_dir(job_id: str) -> Path:
    return RUNS_DIR / job_id


def current_job() -> dict | None:
    """지금 도는 작업. 없으면 `None`."""
    marker = ROOT / "current"
    try:
        job_id = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not job_id:
        return None
    return read_json(run_dir(job_id) / "job.json")


def set_current(job_id: str | None) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "current").write_text(job_id or "", encoding="utf-8")


def current_side() -> dict | None:
    """지금 도는 곁다리. ``current_side`` 마커는 queue의 ``current``와 같은 모양이다."""
    try:
        job_id = CURRENT_SIDE_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not job_id:
        return None
    return read_json(run_dir(job_id) / "job.json")


def set_current_side(job_id: str | None) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    CURRENT_SIDE_FILE.write_text(job_id or "", encoding="utf-8")


def session_alive(name: str) -> bool:
    try:
        return subprocess.run(
            ["tmux", "has-session", "-t", name], capture_output=True
        ).returncode == 0
    except OSError:
        return False


def train_sessions(
    current_session: str | None = None, *, tmux_socket: str | None = None,
) -> list[str] | None:
    """다른 문에서 시작한 ``train-*`` tmux 세션.

    CUDA 초기화 전의 학습은 nvidia-smi에 수십 초 동안 보이지 않을 수 있지만 tmux 세션은
    먼저 생긴다. tmux 서버 자체가 없는 것은 세션이 없는 정상 상태다. 그 밖에 세션 목록을
    못 읽었으면 ``None``이다. 안전하다는 것을 확인하지 못한 상태를 빈 목록으로 바꾸면 바로
    두 학습을 겹쳐 띄울 수 있다. ``tmux_socket``은 격리 소켓으로 이 경로를 시험할 때 쓴다.
    """
    command = ["tmux"]
    if tmux_socket is not None:
        command.extend(["-L", tmux_socket])
    command.extend(["list-sessions", "-F", "#{session_name}"])
    try:
        out = subprocess.run(
            command,
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        if "no server running" in out.stderr or "error connecting" in out.stderr:
            return []
        return None
    return [
        name for name in (line.strip() for line in out.stdout.splitlines())
        if name.startswith("train-") and name != current_session
    ]


def session_process_ids(session: str) -> set[str] | None:
    """tmux 세션의 pane과 그 모든 자손 PID. 조회하지 못하면 필터링하지 않도록 ``None``."""
    try:
        out = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    seed = [line.strip() for line in out.stdout.splitlines() if line.strip().isdigit()]
    return probe.child_pids(seed)


def without_session_apps(apps: list[dict], session: str) -> list[dict]:
    """세션 프로세스 트리에 속한 GPU 프로세스를 제외한다. 조회 실패면 안전하게 원본 그대로."""
    pids = session_process_ids(session)
    if pids is None:
        return apps
    return [app for app in apps if str(app.get("pid")) not in pids]


def compute_apps() -> list[dict] | None:
    """GPU에 붙어 있는 컴퓨트 프로세스. 이 기계가 그것을 볼 수 있을 때만 목록이 온다.

    **못 읽은 것과 없는 것을 가른다.** 잠깐의 실패로 온 빈 목록을 `GPU가 비었다`로 읽는
    순간 큐는 남이 돌리는 학습 위에 하나 더 띄운다 — 통합메모리라 그때 둘 다 스왑으로
    느려진다. 그래서 실패는 `None`이고, 그것을 받은 쪽은 다음 것을 꺼내지 않고 기다린다.

    맥에는 이 질문에 답하는 것이 아예 없어 늘 `None`이다. 그 기계에서는 문지기가 이
    검사를 건너뛴다 — `WATCHES_GPU_PROCESSES`를 보라.
    """
    return probe.gpu_processes()


# ---------------------------------------------------------------- 진행 읽기

_SUFFIX = {"": 1, "K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}
_LEROBOT_STEP = re.compile(r"step:([0-9.]+)([KMBT]?)")
_LEROBOT_LOSS = re.compile(r"loss:([0-9.eE+-]+)")
#: tqdm 막대. `123/20000 [01:02<2:45:10,  2.01it/s]` — 대괄호 안의 `<` 뒤가 tqdm이 스스로 센
#: 남은 시간이다. 막대가 다른 모양이어도 `N/M [`까지는 같으므로 뒤쪽은 선택이다.
_TQDM = re.compile(r"(\d+)/(\d+) \[(?:[\d:]+<([\d:?]+))?")
#: 그 가운데 **학습** 막대. lerobot은 학습 말고도 tqdm을 쓴다 — SmolVLA는 시작할 때
#: `Loading weights: 489/489`을 찍는데, 그것을 학습 진행으로 읽으면 화면이 시작하자마자
#: `100%`가 된다. 사람이 그것을 보고 끝난 줄 알 자리는 아니다.
_LEROBOT_BAR = re.compile(r"Training:.*?(\d+)/(\d+) \[(?:[\d:]+<([\d:?]+))?")
# tqdm은 1 step/s를 경계로 단위를 뒤집는다. API에서는 비교 가능한 초/스텝 하나로 통일한다.
_TQDM_RATE = re.compile(r"([0-9.]+)\s*(step/s|s/step)")
#: lerobot의 `step:` 줄에 실린 스텝당 시간. tqdm이 아직 남은 시간을 모를 때(`?`)의 대안이다.
_LEROBOT_SECS = re.compile(r"updt_s:([0-9.eE+-]+).*?data_s:([0-9.eE+-]+)")
#: 홀드아웃 검증 손실. `--dataset.eval_split`을 켠 학습만 찍는다.
_LEROBOT_EVAL = re.compile(r"step (\d+): eval_loss=([0-9.eE+-]+)")
_RSL_ITER = re.compile(r"Learning iteration (\d+)/(\d+)")
_RSL_ETA = re.compile(r"ETA:\s+(\d+):(\d+):(\d+)")
_RSL_REWARD = re.compile(r"Mean reward:\s+([-0-9.]+)")


def _clock_seconds(text: str | None) -> int | None:
    """tqdm의 `1:02:03`·`02:03`을 초로. `?`(아직 모름)와 빈 값은 `None`."""
    if not text or "?" in text:
        return None
    try:
        parts = [int(piece) for piece in text.split(":")]
    except ValueError:
        return None
    seconds = 0
    for piece in parts:
        seconds = seconds * 60 + piece
    return seconds


def tail_lines(path: Path, count: int = 400) -> list[str]:
    """로그 꼬리. 몇 시간짜리 학습의 로그는 수십 MB가 되므로 끝에서만 읽는다."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - 200_000))
            chunk = handle.read()
    except OSError:
        return []
    text = chunk.decode("utf-8", errors="replace")
    return text.splitlines()[-count:]


def log_lines(path: Path, chunk: int = 1 << 20):
    """로그를 처음부터 한 줄씩. tqdm이 `\r`로만 갱신하므로 `\n`만으로 자르면 안 된다.

    `\n` 하나 없이 수십 MB가 이어지는 구간이 실제로 생긴다(매 스텝 갱신되는 막대). 파일을
    통째로 읽어 `splitlines()`를 부르면 그 한 줄이 메모리에 그대로 올라오므로, 조각으로
    읽어 두 문자 모두에서 자른다.
    """
    carry = ""
    try:
        with path.open("rb") as handle:
            while True:
                raw = handle.read(chunk)
                if not raw:
                    break
                text = carry + raw.decode("utf-8", errors="replace")
                pieces = text.replace("\r", "\n").split("\n")
                carry = pieces.pop()
                yield from pieces
    except OSError:
        return
    if carry:
        yield carry


def thin(points: list[list[float]], limit: int) -> list[list[float]]:
    """점이 너무 많으면 고르게 솎되 **마지막 점은 반드시 남긴다.**

    마지막이 지금 값이다. 그것을 솎아 내면 화면의 곡선 끝과 옆에 적힌 숫자가 서로 다른
    것을 말하게 된다.
    """
    if len(points) <= limit:
        return points
    stride = len(points) / limit
    picked = [points[int(index * stride)] for index in range(limit)]
    if picked[-1] is not points[-1]:
        picked[-1] = points[-1]
    return picked


def series(job_id: str, limit: int = 400) -> dict:
    """작업 하나가 남긴 값의 흐름.

    진행률은 **얼마나 왔나**를 말하지만 **되고 있나**는 말하지 않는다. 밤새 돌린 것을
    아침에 보고 다음을 정하려면 그 곡선이 있어야 한다 — 손실이 내려가다 멈췄는지,
    보상이 오르기 시작했는지는 마지막 숫자 하나로는 보이지 않는다.

    값은 로그에서만 읽는다. 데몬이 따로 적어 두면 재시작 전후로 끊긴 곡선이 남고, 그때
    어느 쪽이 사실인지 정해야 한다. 로그는 tmux 안의 작업이 직접 쓴 것이라 늘 이어져 있다.
    """
    job = read_json(run_dir(job_id) / "job.json")
    if job is None:
        raise Missing(job_id)
    flavour = job.get("progress", "none")
    path = run_dir(job_id) / "run.log"
    train: list[list[float]] = []
    held_out: list[list[float]] = []

    if flavour == "lerobot":
        for line in log_lines(path):
            evaluated = _LEROBOT_EVAL.search(line)
            if evaluated is not None:
                try:
                    held_out.append([float(evaluated.group(1)), float(evaluated.group(2))])
                except ValueError:
                    pass
                continue
            if "loss:" not in line:
                continue
            step, loss = _LEROBOT_STEP.search(line), _LEROBOT_LOSS.search(line)
            if step is None or loss is None:
                continue
            try:
                train.append([
                    float(step.group(1)) * _SUFFIX.get(step.group(2), 1),
                    float(loss.group(1)),
                ])
            except ValueError:
                pass
    elif flavour == "rsl_rl":
        iteration = None
        for line in log_lines(path):
            found = _RSL_ITER.search(line)
            if found is not None:
                iteration = int(found.group(1))
                continue
            if iteration is None:
                continue
            reward = _RSL_REWARD.search(line)
            if reward is not None:
                try:
                    train.append([float(iteration), float(reward.group(1))])
                except ValueError:
                    pass

    out = []
    if train:
        name, label, axis = (
            ("reward", "평균 보상", "반복") if flavour == "rsl_rl" else ("loss", "학습 손실", "스텝")
        )
        out.append({"name": name, "label": label, "axis": axis, "points": thin(train, limit)})
    if held_out:
        out.append({"name": "eval_loss", "label": "검증 손실", "axis": "스텝",
                    "points": thin(held_out, limit)})
    return {"id": job_id, "progress": flavour, "series": out}


def parse_progress(flavour: str, lines: list[str]) -> dict:
    """종류마다 다른 로그를 화면이 쓰는 한 가지 모양으로 옮긴다."""
    found: dict[str, object] = {}
    if flavour == "lerobot":
        # 스텝 수의 주인은 tqdm 막대다. `step:` 줄은 loss와 스텝당 시간을 주고, 막대가
        # 아직 없을 때만 스텝의 대역이 된다.
        #
        # 전에는 둘 중 **큰 쪽**을 썼다. 같은 것을 세는 두 계량이라고 보았기 때문인데,
        # 이어붙인 학습에서 그 가정이 깨진다 — `--resume`으로 5,000스텝을 더 돌리면
        # tqdm은 이번 구간만 `0/5000`으로 세고 `step:` 줄은 통산 `5100`을 찍는다. 큰
        # 쪽을 고르면 5,000짜리 막대에 5,100이 들어가 진행률이 100%를 넘는다.
        #
        # 이번 구간을 세는 쪽이 맞다. `이어서 한 밤 더`를 건 사람이 알고 싶은 것은 통산
        # 스텝이 아니라 오늘 밤이 어디까지 갔는가다.
        for line in reversed(lines):
            hit = _LEROBOT_STEP.search(line)
            if hit:
                try:
                    found["_step"] = int(round(float(hit.group(1)) * _SUFFIX.get(hit.group(2), 1)))
                except ValueError:
                    pass
                loss = _LEROBOT_LOSS.search(line)
                if loss:
                    try:
                        found["loss"] = float(loss.group(1))
                    except ValueError:
                        pass
                secs = _LEROBOT_SECS.search(line)
                if secs:
                    try:
                        found["_step_seconds"] = float(secs.group(1)) + float(secs.group(2))
                    except ValueError:
                        pass
                break
        for line in reversed(lines):
            bar = _LEROBOT_BAR.search(line)
            if bar:
                found["step"], found["steps"] = int(bar.group(1)), int(bar.group(2))
                # 남은 시간은 tqdm이 센 것을 그대로 쓴다. 이 데몬이 스텝 속도를 따로 재면
                # 같은 것을 세는 계량이 둘이 되고, 둘이 어긋나는 날 어느 쪽을 믿을지 정해야 한다.
                remaining = _clock_seconds(bar.group(3))
                if remaining is not None:
                    found["eta_seconds"] = remaining
                rate = _TQDM_RATE.search(line)
                if rate:
                    try:
                        value = float(rate.group(1))
                        if value > 0:
                            found["step_seconds"] = 1.0 / value if rate.group(2) == "step/s" else value
                    except ValueError:
                        pass
                break
        # tqdm이 아직 남은 시간을 모르거나(첫 몇 스텝은 `?`) 막대가 없으면, `step:` 줄의
        # 스텝당 시간으로 센다. 그것마저 없으면 남은 시간을 지어내지 않는다.
        fallback = found.pop("_step", None)
        if "step" not in found and fallback is not None:
            found["step"] = fallback
        step_seconds = found.pop("_step_seconds", None)
        if "step_seconds" not in found and step_seconds:
            found["step_seconds"] = step_seconds
        if "eta_seconds" not in found and step_seconds and found.get("steps") and found.get("step") is not None:
            found["eta_seconds"] = int(max(0, found["steps"] - found["step"]) * step_seconds)
    elif flavour == "rsl_rl":
        for line in reversed(lines):
            hit = _RSL_ITER.search(line)
            if hit:
                found["step"], found["steps"] = int(hit.group(1)), int(hit.group(2))
                break
        for line in reversed(lines):
            hit = _RSL_ETA.search(line)
            if hit:
                hours, minutes, seconds = (int(piece) for piece in hit.groups())
                found["eta_seconds"] = hours * 3600 + minutes * 60 + seconds
                break
        for line in reversed(lines):
            hit = _RSL_REWARD.search(line)
            if hit:
                try:
                    found["reward"] = float(hit.group(1))
                except ValueError:
                    pass
                break
    return found


def progress_of(job: dict) -> dict:
    """도는 작업의 진행. 로그를 읽어 만들고, 끝난 작업은 마지막으로 적힌 것을 그대로 쓴다."""
    directory = run_dir(job["id"])
    lines = tail_lines(directory / "run.log")
    found = parse_progress(job.get("progress", "none"), lines)
    # 총 스텝을 로그가 말해 주지 않는 종류도 있다(lerobot의 `--steps`). 걸 때 알고 있던
    # 값이 있으면 그것을 쓴다.
    if "steps" not in found:
        declared = (job.get("values") or {}).get("steps")
        if isinstance(declared, int):
            found["steps"] = declared
    found["log_tail"] = [line for line in lines[-6:] if line.strip()]
    try:
        found["updated_at"] = (directory / "run.log").stat().st_mtime
    except OSError:
        found["updated_at"] = None
    return found


# ---------------------------------------------------------------- 실행기

def start(job: dict) -> dict:
    """tmux 안에서 띄운다. 데몬이 프로세스를 직접 품지 않는다.

    데몬은 `systemctl --user restart` 한 번에 다시 시작되는 서비스이고, 학습은 몇 시간
    돈다. tmux 안에 있으면 데몬이 내려갔다 올라와도 학습은 그대로 돌고, 올라온 데몬이
    세션 이름으로 그것을 다시 찾는다.
    """
    directory = run_dir(job["id"])
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "run.sh"
    # 명령을 파일로 떨어뜨리는 이유: tmux에 넘길 문자열 안에서 따옴표를 다시 escape 하지
    # 않아도 되고, **무엇이 실제로 돌았는지를 사람이 그대로 읽을 수 있다.**
    log = directory / "run.log"
    script.write_text(run_script_text(job["command"], log), encoding="utf-8")
    script.chmod(0o755)
    code = directory / "exit_code"
    code.unlink(missing_ok=True)
    (directory / "cancelled").unlink(missing_ok=True)
    # 감싸는 자리를 `run.sh` 안이 아니라 여기로 둔 이유가 있다. 그 파일은 **무엇이 돌
    # 것인가**를 사람이 그대로 읽는 자리이고, 거기에 기계 사정(`caffeinate`)이 섞이면
    # 읽는 사람이 명령과 시중드는 것을 갈라 읽어야 한다.
    line = f"{probe.wrap(f'bash {script}')} > {log} 2>&1; echo $? > {code}"
    subprocess.run(["tmux", "new", "-d", "-s", job["session"], line], check=True, timeout=60)
    job["state"] = "running"
    job["started_at"] = time.time()
    write_json(directory / "job.json", job)
    set_current(job["id"])
    return job


def start_side(kind: str, params: dict) -> dict:
    """빈 곁다리 슬롯에 짧은 작업을 즉시 띄운다."""
    with QUEUE_LOCK:
        return _start_side(kind, params)


def _start_side(kind: str, params: dict) -> dict:
    if current_side() is not None:
        raise Conflict("이미 도는 곁다리가 있습니다")
    kinds = load_kinds()
    spec = kinds.get(kind)
    if spec is None:
        raise Invalid(f"알 수 없는 작업 종류입니다: {kind}")
    if spec["lane"] != "side":
        raise Invalid(f"큐 종류는 /api/queue로 걸어야 합니다: {kind}")
    # 시한이 이 레인의 전부다. 강제할 수 없으면 띄우지 않는 것이 맞다 — 사고의 원인은
    # "누가 껐다"가 아니라 아무도 안 껐다는 것이었고, 그것을 막는 것이 이 한 줄이다.
    limiter = probe.timeout_prefix(MAX_SIDE_SECONDS)
    if limiter is None:
        raise Invalid(
            "이 기계에는 시한을 강제할 timeout이 없어 곁다리를 띄우지 않습니다 "
            "(맥이라면 `brew install coreutils`로 gtimeout을 깔면 됩니다)"
        )

    values = validate(spec, params or {})
    job_id = new_id()
    full = expand(spec, values, job_id)
    training = current_job()
    baseline = None
    if training is not None and session_alive(training.get("session", "")):
        baseline = progress_of(training).get("step_seconds")
    started = time.time()
    job = {
        "id": job_id,
        "kind": kind,
        "lane": "side",
        "title": Template(spec.get("title", kind)).safe_substitute(full),
        "params": {key: values[key] for key in (f["name"] for f in spec.get("fields", [])) if key in values},
        "values": values,
        "command": Template(spec["run"]).safe_substitute(full),
        "session": _session_name(spec, full),
        "limit_seconds": spec["limit_seconds"],
        "state": "running",
        "started_at": started,
        "expires_at": started + spec["limit_seconds"],
        "extendable_until": started + MAX_SIDE_SECONDS,
        "baseline_step_seconds": baseline,
    }

    directory = run_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    command_script = directory / "side-command.sh"
    log = directory / "run.log"
    command_script.write_text(run_script_text(job["command"], log), encoding="utf-8")
    command_script.chmod(0o755)
    script = directory / "run.sh"
    # 최초 만료는 데몬이, 연장 가능한 절대 상한은 이 timeout도 함께 지킨다. 최초 600초로
    # 감싸면 API로 10분을 연장해도 먼저 죽으므로 wrapper에는 extendable_until의 상한을 쓴다.
    script.write_text(
        "#!/usr/bin/env bash\nset -o pipefail\n"
        f"exec {limiter} bash {shlex.quote(str(command_script))}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    code = directory / "exit_code"
    code.unlink(missing_ok=True)
    for marker in (directory / "cancelled", directory / "expired"):
        marker.unlink(missing_ok=True)
    write_json(directory / "job.json", job)
    set_current_side(job_id)
    line = (
        f"{probe.wrap(f'bash {shlex.quote(str(script))}')}"
        f" > {shlex.quote(str(log))} 2>&1; echo $? > {shlex.quote(str(code))}"
    )
    try:
        subprocess.run(["tmux", "new", "-d", "-s", job["session"], line], check=True, timeout=60)
    except Exception:
        job["state"] = "failed"
        job["finished_at"] = time.time()
        write_json(directory / "job.json", job)
        set_current_side(None)
        raise
    return side_view(job, training)


def finalize(job: dict) -> dict:
    """세션이 사라진 작업을 정리한다. 왜 끝났는지를 남기는 것이 요점이다."""
    directory = run_dir(job["id"])
    raw = None
    try:
        raw = int((directory / "exit_code").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    if (directory / "cancelled").exists():
        job["state"] = "cancelled"
    elif raw == 0:
        job["state"] = "done"
    elif raw is None:
        # 종료 코드조차 남기지 못했다. 세션이 밖에서 죽었거나 기계가 재부팅됐다.
        job["state"] = "interrupted"
    else:
        job["state"] = "failed"
    job["exit_code"] = raw
    job["finished_at"] = time.time()
    job["progress"] = job.get("progress", "none")
    write_json(directory / "progress.json", progress_of(job))
    write_json(directory / "job.json", job)
    set_current(None)
    return job


def finalize_side(job: dict) -> dict:
    """끝난 곁다리를 기록하고 전용 마커만 비운다."""
    directory = run_dir(job["id"])
    raw = None
    try:
        raw = int((directory / "exit_code").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    if (directory / "expired").exists() or raw == 124:
        job["state"] = "expired"
    elif (directory / "cancelled").exists():
        job["state"] = "cancelled"
    elif raw == 0:
        job["state"] = "done"
    elif raw is None:
        job["state"] = "interrupted"
    else:
        job["state"] = "failed"
    job["exit_code"] = raw
    job["finished_at"] = time.time()
    write_json(directory / "job.json", job)
    set_current_side(None)
    return job


def tick() -> None:
    """한 박자. 상태 전환을 DELETE와 겹치지 않게 살핀다."""
    with QUEUE_LOCK:
        _tick()


def _tick() -> None:
    """도는 것을 살피고, GPU와 학습 세션이 모두 비어 있으면 다음 것을 꺼낸다."""
    side = current_side()
    if side is not None:
        if not session_alive(side.get("session", "")):
            finalize_side(side)
            side = None
        elif time.time() >= float(side.get("expires_at") or 0):
            _stop_side(expired=True)
            side = None

    job = current_job()
    if job is not None:
        if session_alive(job.get("session", "")):
            write_json(run_dir(job["id"]) / "progress.json", progress_of(job))
            return
        finalize(job)
        return

    if PAUSED_FILE.exists():
        return
    files = queued_files()
    if not files:
        return
    # CUDA 초기화 전에는 train 세션이 이미 있어도 nvidia-smi에 보이지 않는다. 어느 쪽이든
    # 다른 작업이 있으면 기다리고, `None`(상태를 못 읽음)도 안전하다고 추측하지 않는다.
    sessions = train_sessions()
    if sessions is None or sessions:
        return
    # GPU 검사는 그것을 볼 수 있는 기계에서만 한다. 맥에는 이 질문에 답하는 것이 없고,
    # 있었어도 쓸 수 없다 — 그 기계에서는 앱 자신이 늘 GPU를 쓰므로 "비어야 시작한다"를
    # 옮겨 오면 큐가 영영 시작하지 않는다. 겹치면 안 되는 것은 학습 둘이고, 학습을 띄우는
    # 문은 이 큐 하나이므로 위의 세션 검사가 그 자리를 대신한다.
    if WATCHES_GPU_PROCESSES:
        apps = compute_apps()
        if apps is not None and side is not None:
            apps = without_session_apps(apps, side.get("session", ""))
        if apps is None or apps:
            return
    job = read_json(files[0])
    if job is None:
        files[0].unlink(missing_ok=True)
        return
    # 큐 파일을 지우기 **전에** 실행 자리에 job.json을 남긴다. 지운 뒤에 start()가 예상 밖의
    # 예외로 죽으면, 큐에도 없고 runs/에도 없는 작업이 되어 조용히 증발한다 — 자는 동안
    # 도는 것들이므로 가장 나쁜 결말이다.
    job["state"] = "running"
    write_json(run_dir(job["id"]) / "job.json", job)
    files[0].unlink(missing_ok=True)
    try:
        start(job)
    except Exception as error:  # noqa: BLE001 — 어떤 이유로 못 띄우든 작업은 failed로 남아야 한다
        job["state"] = "failed"
        job["error"] = f"작업을 띄우지 못했습니다: {error}"
        job["finished_at"] = time.time()
        write_json(run_dir(job["id"]) / "job.json", job)
        set_current(None)


def reconcile() -> None:
    """데몬이 올라올 때 한 번. 도는 것으로 적혀 있는데 세션이 없으면 정리한다."""
    job = current_job()
    if job is not None and not session_alive(job.get("session", "")):
        finalize(job)
    side = current_side()
    if side is not None and not session_alive(side.get("session", "")):
        finalize_side(side)


def stop(job_id: str) -> dict:
    """도는 작업을 세우거나, 아직 대기 중인 작업을 줄에서 뺀다.

    먼저 `C-c`인 것은 그것이 사람이 tmux에 붙어 눌렀을 때와 같은 길이기 때문이다 —
    LeRobot은 그 신호를 받고 정리한 뒤 나가므로 지금까지의 체크포인트가 온전히 남는다.
    """
    with QUEUE_LOCK:
        return _stop(job_id)


def _stop(job_id: str) -> dict:
    """tick의 꺼내기와 한 락 안에서 실행하는 실제 중지 동작."""
    path = find_queued(job_id)
    if path is not None:
        job = read_json(path) or {"id": job_id}
        path.unlink(missing_ok=True)
        job["state"] = "cancelled"
        job["finished_at"] = time.time()
        write_json(run_dir(job_id) / "job.json", job)
        return job
    job = current_job()
    if job is None or job.get("id") != job_id:
        # 도는 것도 대기도 아니다. 방금 자연 종료했을 수 있다 — 그때는 404 대신 그 결과를
        # 돌려준다. `중지`를 누른 찰나에 끝나는 것은 실패가 아니다.
        existing = read_json(run_dir(job_id) / "job.json")
        if existing is not None:
            return existing
        raise Missing(job_id)
    (run_dir(job_id) / "cancelled").write_text("1", encoding="utf-8")
    session = job.get("session", "")
    subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], capture_output=True, timeout=30)
    time.sleep(2.0)
    if session_alive(session):
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, timeout=30)
    return finalize(current_job() or job)


def stop_side() -> dict:
    with QUEUE_LOCK:
        return _stop_side()


def _stop_side(*, expired: bool = False) -> dict:
    job = current_side()
    if job is None:
        raise Missing("곁다리")
    directory = run_dir(job["id"])
    marker = directory / ("expired" if expired else "cancelled")
    marker.write_text("1", encoding="utf-8")
    session = job.get("session", "")
    subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], capture_output=True, timeout=30)
    time.sleep(2.0)
    if session_alive(session):
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, timeout=30)
    return finalize_side(current_side() or job)


def extend_side(seconds: object) -> dict:
    """곁다리 만료를 늘린다. 시작 뒤 한 시간을 넘겨 조용히 장기 작업이 될 수는 없다."""
    if isinstance(seconds, bool):
        raise Invalid("seconds는 양의 정수여야 합니다")
    try:
        amount = int(seconds)
    except (TypeError, ValueError):
        raise Invalid("seconds는 양의 정수여야 합니다") from None
    if amount <= 0 or str(amount) != str(seconds):
        raise Invalid("seconds는 양의 정수여야 합니다")
    with QUEUE_LOCK:
        job = current_side()
        if job is None:
            raise Missing("곁다리")
        expires = float(job["expires_at"]) + amount
        if expires > float(job["extendable_until"]):
            raise Invalid("곁다리는 시작 뒤 3600초를 넘겨 연장할 수 없습니다")
        job["expires_at"] = expires
        write_json(run_dir(job["id"]) / "job.json", job)
        return side_view(job)


def side_view(job: dict | None, training: dict | None = None) -> dict | None:
    """앱 계약에 필요한 곁다리 칸만 내보낸다."""
    if job is None:
        return None
    if training is None:
        training = current_job()
    step_seconds = None
    if training is not None and session_alive(training.get("session", "")):
        step_seconds = progress_of(training).get("step_seconds")
    return {
        "id": job.get("id"),
        "kind": job.get("kind"),
        "title": job.get("title"),
        "session": job.get("session"),
        "started_at": job.get("started_at"),
        "expires_at": job.get("expires_at"),
        "extendable_until": job.get("extendable_until"),
        "baseline_step_seconds": job.get("baseline_step_seconds"),
        "step_seconds": step_seconds,
    }


def recent(limit: int = 20) -> list[dict]:
    """최근에 끝난 것들. 새것부터."""
    if not RUNS_DIR.is_dir():
        return []
    jobs = []
    for directory in RUNS_DIR.iterdir():
        job = read_json(directory / "job.json")
        # 정상 종료한 곁다리는 쌓지 않되, 실패는 앱에서 로그를 열 수 있게 남긴다.
        if (job and job.get("state") not in {"running", "queued"}
                and (job.get("lane") != "side" or job.get("state") == "failed")):
            jobs.append(job)
    jobs.sort(key=lambda item: item.get("finished_at") or 0, reverse=True)
    return jobs[:limit]


def snapshot() -> dict:
    """화면 한 장에 필요한 전부. 왕복을 늘리지 않으려고 한 번에 답한다."""
    job = current_job()
    side = current_side()
    running = None
    if job is not None:
        running = dict(job)
        running["live"] = session_alive(job.get("session", ""))
        running["progress_detail"] = progress_of(job)
    queued = []
    for path in queued_files():
        entry = read_json(path)
        if entry:
            queued.append(entry)
    out: dict[str, object] = {
        "running": running,
        "side": side_view(side, job),
        "queued": queued,
        "recent": recent(),
        "paused": PAUSED_FILE.exists(),
        "foreign_sessions": train_sessions(job.get("session") if job is not None else None),
        "capabilities": capabilities(),
    }
    # 볼 수 없는 기계에서는 이 칸을 **아예 싣지 않는다.** 빈 목록으로 실으면 "확인했고
    # 비어 있다"가 되는데, 실제로는 확인할 방법이 없었던 것이다.
    if WATCHES_GPU_PROCESSES:
        gpu_apps = compute_apps() or []
        if job is not None:
            gpu_apps = without_session_apps(gpu_apps, job.get("session", ""))
        if side is not None:
            gpu_apps = without_session_apps(gpu_apps, side.get("session", ""))
        # 세션의 프로세스 트리를 못 읽었을 때만 안전한 쪽으로 전체 GPU 목록을 그대로 싣는다.
        out["gpu_apps"] = gpu_apps
    return out


def datasets() -> list[dict]:
    """학습에 걸 수 있는 데이터셋. `~/data/soarm` 아래에 와 있는 것들."""
    out = []
    if not DATASET_ROOT.is_dir():
        return out
    for directory in sorted(DATASET_ROOT.iterdir()):
        info = directory / "meta" / "info.json"
        if directory.name.startswith(".") or not info.is_file():
            continue
        meta = read_json(info) or {}
        out.append({
            "name": directory.name,
            "episodes": meta.get("total_episodes", 0),
            "frames": meta.get("total_frames", 0),
            "fps": meta.get("fps", 0),
        })
    return out


def runs() -> list[dict]:
    """이어붙일 수 있는 학습. `~/outputs/*` 가운데 체크포인트가 남아 있는 것들.

    이 목록이 있어야 하는 이유는 밤의 길이 때문이다. 사람이 자는 동안이 일곱에서 여덟
    시간인데, 이 맥에서 SmolVLA 2만 스텝은 27시간이다. 한 밤에 안 끝나는 것을 밤마다
    이어 붙이는 것이 이 기계의 기본 사용법이고(실측으로 이어붙이는 값은 20초다), 그러려면
    무엇을 이어붙일 수 있는지가 목록으로 보여야 한다.

    `checkpoints/last`는 마지막 체크포인트 디렉터리를 가리키는 심볼릭 링크이고, 그 이름이
    곧 지금까지 간 스텝이다. 나머지(목표 스텝·정책·데이터셋)는 그 안의 `train_config.json`
    에 있다 — 이어붙일 때 lerobot이 읽는 파일과 같은 것이라 화면과 실행이 어긋나지 않는다.
    """
    out = []
    if not OUTPUT_ROOT.is_dir():
        return out
    for directory in OUTPUT_ROOT.iterdir():
        if directory.name.startswith(".") or not NAME.match(directory.name):
            continue
        last = directory / "checkpoints" / "last"
        config = last / "pretrained_model" / "train_config.json"
        if not config.is_file():
            continue
        meta = read_json(config) or {}
        try:
            step = int(last.resolve().name)
        except (OSError, ValueError):
            step = 0
        try:
            updated = config.stat().st_mtime
        except OSError:
            updated = 0.0
        out.append({
            "name": directory.name,
            "step": step,
            "steps": meta.get("steps") or 0,
            "policy": (meta.get("policy") or {}).get("type") or "",
            "dataset": (meta.get("dataset") or {}).get("repo_id") or "",
            "updated_at": updated,
        })
    # 최근에 손댄 것이 위로. 밤마다 이어 붙이는 것은 거의 늘 어젯밤 것이다.
    out.sort(key=lambda item: item["updated_at"], reverse=True)
    return out


def source_names(source: str) -> list[str] | None:
    """`name` 칸의 `source`가 가리키는 목록. 모르는 이름이면 `None`이라 검사하지 않는다."""
    if source == "datasets":
        return [item["name"] for item in datasets()]
    if source == "runs":
        return [item["name"] for item in runs()]
    return None


def capabilities() -> dict:
    """이 기계가 무엇을 **확인할 수 있는가.**

    화면이 없는 검사를 `통과`로 그리지 않게 하려고 있다. 맥에서 `gpu_apps`를 빈 목록으로
    실으면 "확인했고 비어 있다"로 읽히는데, 실제로는 확인할 방법이 없었던 것이다. 그 둘은
    사람이 할 일이 다르다 — 앞이면 안심해도 되고, 뒤면 스스로 살펴야 한다.
    """
    return {"platform": probe.KIND, "gpu_processes": WATCHES_GPU_PROCESSES}


def machine() -> dict:
    """기계 상태 한 줌.

    기계마다 다른 칸(GPU·CPU·메모리·전원)은 `probe`가 채우고, 여기서는 큐만 아는 것을
    얹는다. 통합메모리 기계에서는 GPU 전용 메모리 칸이 비는데, 그때는 `memory`가 곧 이
    기계가 얼마나 찼는지다 — GB10도 M2 Max도 CPU와 GPU가 한 메모리를 나눠 쓴다.
    """
    info: dict[str, object] = {"ok": True, "host": os.uname().nodename}
    info.update(probe.machine())
    usage = shutil.disk_usage(str(HOME))
    info["disk_free_bytes"] = usage.free
    info["disk_total_bytes"] = usage.total
    job = current_job()
    info["running"] = bool(job and session_alive(job.get("session", "")))
    info["queued"] = len(queued_files())
    info["paused"] = PAUSED_FILE.exists()
    info["capabilities"] = capabilities()
    return info


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "sparkq"

    def log_message(self, *_args) -> None:  # 요청마다 stderr에 한 줄씩 쌓지 않는다.
        pass

    def _send(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            payload = json.loads(self.rfile.read(length))
        except ValueError:
            raise Invalid("요청 본문이 JSON이 아닙니다") from None
        return payload if isinstance(payload, dict) else {}

    def _route(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        parts = [piece for piece in path.split("/") if piece]
        method = self.command
        if method == "GET" and parts == ["api", "status"]:
            return machine()
        if method == "GET" and parts == ["api", "kinds"]:
            return {"kinds": list(load_kinds().values())}
        if method == "GET" and parts == ["api", "datasets"]:
            return {"datasets": datasets()}
        if method == "GET" and parts == ["api", "runs"]:
            return {"runs": runs()}
        if method == "GET" and parts == ["api", "queue"]:
            return snapshot()
        if method == "POST" and parts == ["api", "side"]:
            payload = self._body()
            return start_side(str(payload.get("kind", "")), payload.get("params") or {})
        if method == "DELETE" and parts == ["api", "side"]:
            return stop_side()
        if method == "POST" and parts == ["api", "side", "extend"]:
            return extend_side(self._body().get("seconds"))
        if method == "POST" and parts == ["api", "queue"]:
            payload = self._body()
            return enqueue(str(payload.get("kind", "")), payload.get("params") or {})
        if method == "POST" and parts == ["api", "queue", "pause"]:
            paused = bool(self._body().get("paused", True))
            ROOT.mkdir(parents=True, exist_ok=True)
            if paused:
                PAUSED_FILE.write_text("1", encoding="utf-8")
            else:
                PAUSED_FILE.unlink(missing_ok=True)
            return {"paused": paused}
        if len(parts) >= 3 and parts[:2] == ["api", "queue"]:
            job_id = parts[2]
            if not NAME.match(job_id):
                raise Invalid("작업 번호 형식이 아닙니다")
            if method == "DELETE" and len(parts) == 3:
                return stop(job_id)
            if method == "POST" and parts[3:] == ["top"]:
                return move_to_top(job_id)
            if method == "GET" and parts[3:] == ["log"]:
                return {"id": job_id, "lines": tail_lines(run_dir(job_id) / "run.log", 400)}
            if method == "GET" and parts[3:] == ["series"]:
                return series(job_id)
        return None

    def _dispatch(self) -> None:
        try:
            result = self._route()
        except Invalid as error:
            self._send({"detail": str(error)}, 400)
        except Missing as error:
            self._send({"detail": f"그런 작업이 없습니다: {error}"}, 404)
        except Conflict as error:
            self._send({"detail": str(error)}, 409)
        except Exception as error:  # 데몬이 요청 하나로 죽지 않게 한다.
            self._send({"detail": f"{type(error).__name__}: {error}"}, 500)
        else:
            if result is None:
                self._send({"detail": "없는 경로입니다"}, 404)
            else:
                self._send(result)

    do_GET = do_POST = do_DELETE = _dispatch


def daemon() -> None:
    for directory in (ROOT, QUEUE_DIR, RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    # 포트를 **루프 전에** 잡는다. 둘째 데몬이 뜨면 여기서 바인딩에 실패해 즉시 죽는다.
    # serve()를 스레드에 미뤄 두면 바인딩이 실패해도 그 스레드만 죽고 tick 루프는 계속
    # 돌아, 두 루프가 같은 큐에서 작업을 이중으로 꺼내는 길이 열린다.
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    reconcile()
    threading.Thread(target=server.serve_forever, daemon=True, name="sparkq-http").start()
    print(f"[sparkq] 127.0.0.1:{PORT} 에서 듣습니다. 상태는 {ROOT}", flush=True)
    while True:
        try:
            tick()
        except Exception as error:  # 한 박자가 실패해도 다음 박자는 온다.
            print(f"[sparkq] tick 실패: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        time.sleep(3)


# ---------------------------------------------------------------- 명령줄

def call(method: str, path: str, payload: dict | None = None):
    """CLI도 HTTP를 지난다. 앱과 터미널이 서로 다른 길로 큐를 고치면 언젠가 어긋난다."""
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = json.loads(error.read() or b"{}").get("detail", str(error))
        raise SystemExit(f"sparkq: {detail}") from None
    except urllib.error.URLError:
        raise SystemExit(
            "sparkq: 데몬에 닿지 못했습니다. `systemctl --user status sparkq`를 보세요."
        ) from None


def human(seconds: float | None) -> str:
    if not seconds:
        return "-"
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def show(snap: dict) -> None:
    running = snap.get("running")
    if running:
        detail = running.get("progress_detail") or {}
        step, steps = detail.get("step"), detail.get("steps")
        bar = f"{step}/{steps}" if step and steps else "시작하는 중"
        eta = human(detail.get("eta_seconds"))
        print(f"▶ {running['id']}  {running['title']}  {bar}  남은 시간 {eta}")
    else:
        print("▶ 도는 작업 없음" + ("  (일시정지됨)" if snap.get("paused") else ""))
        for app in snap.get("gpu_apps") or []:
            print(f"   GPU를 큐 밖의 프로세스가 쓰고 있습니다: pid {app['pid']} {app['name']}")
    for index, job in enumerate(snap.get("queued") or [], start=1):
        print(f"{index:2d}. {job['id']}  {job['title']}")
    for job in (snap.get("recent") or [])[:5]:
        mark = {"done": "✓", "failed": "✗", "cancelled": "-", "interrupted": "!"}.get(job.get("state"), "?")
        print(f" {mark} {job['id']}  {job['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sparkq", description="Spark GPU 작업 큐")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("daemon", help="큐 데몬으로 돈다 (systemd가 부른다)")
    sub.add_parser("ls", help="지금 도는 것과 대기열")
    sub.add_parser("kinds", help="걸 수 있는 작업 종류")
    add = sub.add_parser("add", help="작업을 큐에 건다")
    add.add_argument("kind")
    add.add_argument("params", nargs="*", help="이름=값")
    remove = sub.add_parser("rm", help="대기 취소 또는 도는 작업 중지")
    remove.add_argument("id")
    top = sub.add_parser("top", help="맨 앞으로")
    top.add_argument("id")
    log = sub.add_parser("log", help="로그 꼬리")
    log.add_argument("id")
    sub.add_parser("pause", help="다음 작업을 꺼내지 않는다")
    sub.add_parser("resume", help="다시 꺼낸다")
    args = parser.parse_args()

    if args.command == "daemon":
        daemon()
    elif args.command == "ls":
        show(call("GET", "/api/queue"))
    elif args.command == "kinds":
        for spec in call("GET", "/api/kinds")["kinds"]:
            fields = ", ".join(
                f"{field['name']}={'|'.join(map(str, field.get('values', []))) or field.get('type', 'name')}"
                for field in spec.get("fields", [])
            )
            print(f"{spec['kind']}: {fields}")
    elif args.command == "add":
        params = {}
        for pair in args.params:
            if "=" not in pair:
                raise SystemExit(f"sparkq: 이름=값 형태여야 합니다: {pair}")
            key, value = pair.split("=", 1)
            params[key] = value
        job = call("POST", "/api/queue", {"kind": args.kind, "params": params})
        print(f"걸었습니다: {job['id']}  {job['title']}")
    elif args.command == "rm":
        job = call("DELETE", f"/api/queue/{args.id}")
        print(f"{job['id']}: {job['state']}")
    elif args.command == "top":
        job = call("POST", f"/api/queue/{args.id}/top")
        print(f"맨 앞으로: {job['id']}")
    elif args.command == "log":
        for line in call("GET", f"/api/queue/{args.id}/log")["lines"]:
            print(line)
    elif args.command in {"pause", "resume"}:
        state = call("POST", "/api/queue/pause", {"paused": args.command == "pause"})
        print("일시정지" if state["paused"] else "재개")


if __name__ == "__main__":
    main()
