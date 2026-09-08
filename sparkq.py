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

## 왜 비선점인가 — 그리고 "멈춰 두기"는 왜 선점이 아닌가

학습은 몇 시간짜리이고 중간에 뺏으면 처음부터 다시 해야 한다. 그래서 한 번 시작한 작업은
끝나거나 사람이 세울 때까지 둔다. 대신 **아직 시작하지 않은** 줄은 얼마든지 다시 세울 수
있게 한다(`top`, `rm`).

여기에 예외가 하나 있는데, 그것은 **뺏는 것이 아니라 멈춰 두는 것**이다. 곁다리 종류가
`preempt`를 선언하면, 그 곁다리가 도는 동안 학습 프로세스에 `SIGSTOP`을 보내고 끝나면
`SIGCONT`로 깨운다. 위 문단이 걱정하는 "처음부터 다시"가 일어나지 않는다 — 스텝도
옵티마이저 상태도 그대로고, 깨어난 자리가 멈춘 자리다. 통합메모리라 얼어붙은 학습이 쥔
몇십 GB를 그대로 둔 채 GPU **연산만** 비켜 주는 것이 가능하고, 그래서 이 값싼 양보가
성립한다.

무엇이 양보하는지는 비대칭이 정한다. 학습은 재개할 수 있고 아무도 보고 있지 않다. 사람이
팔 앞에 서서 하는 추론은 재개할 수 없고, 옆에서 학습이 돌면 **지연이 흔들려** 결과가
오염된다. 그래서 양보는 학습이 한다. 자세한 것은 `docs/2026-09-07-preempt.md`.

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
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
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
TRASH_DIR = ROOT / "trash"
"""내역에서 뺀 끝난 작업의 기록(job.json, run.log)이 가는 곳. 지우지 않고 옮긴다 — 되돌릴 수 있게."""
PAUSED_FILE = ROOT / "paused"
#: 지금 얼려 둔 학습의 표. 곁다리가 `preempt`로 학습을 멈추면 여기 남고, 깨우면 지운다.
#:
#: **파일이어야 한다.** 얼린 사실을 프로세스 안에만 두면 그 프로세스가 죽는 순간 학습은
#: 영원히 멈춘 채가 된다 — 이 큐에서 가장 무서운 고장이다. 디스크에 두면 데몬이 다시
#: 올라올 때 `reconcile()`이 읽어서 깨운다.
PREEMPTED_FILE = ROOT / "preempted"
KINDS_DIR = Path(os.environ.get("SPARKQ_KINDS", Path(__file__).resolve().parent / "kinds"))
DATASET_ROOT = Path(os.environ.get("SPARKQ_DATASETS", HOME / "data" / "soarm"))
#: 학습이 결과를 쌓는 곳. `lerobot-resume`이 이어붙일 실행을 여기서 찾는다.
OUTPUT_ROOT = Path(os.environ.get("SPARKQ_OUTPUTS", HOME / "outputs"))

#: 학습 실행 **옆**에 로그와 메타를 두는 자리. `<outputs>/.runs/<run>/`.
#:
#: `output_dir` 안에 둘 수 없다. lerobot은 `output_dir`이 이미 있으면 `FileExistsError`로
#: 거절하는데, 로그를 그 안에 두려면 `tee`가 쓸 폴더를 미리 만들어야 하고 그 `mkdir`이
#: 곧 거절 조건이 된다. 점으로 시작하므로 실행 목록에서 실행 이름으로 읽히지도 않는다.
#:
#: **`kinds/*.json`의 `derived.side` 템플릿과 같은 값이어야 한다.** 종류 파일이 만드는
#: 자리를 여기서 지우기 때문이고, 한쪽만 고치면 지운 뒤에도 로그가 남는다.
RUN_SIDE_DIR = ".runs"
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

#: 중지 신호를 준 뒤 작업이 **스스로 정리할** 시간. 종류가 `stop_grace_seconds`로 늘린다.
#:
#: 컨테이너 안에서 도는 작업은 이 시간 안에 자기 자식을 거둬야 한다. `docker exec`는
#: 클라이언트가 죽어도 컨테이너 안으로 신호를 보내지 않으므로, 큐가 세션을 죽이는 것만으로는
#: GPU가 비지 않는다.
DEFAULT_STOP_GRACE = 10.0
#: 정리를 기다리는 동안 큐의 락을 쥐고 있다. 종류 하나가 중지를 영영 붙들 수는 없다.
MAX_STOP_GRACE = 60.0
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
        preempt = spec.get("preempt", False)
        if not isinstance(preempt, bool):
            print(f"[sparkq] 종류 파일을 읽지 않습니다 {path}: preempt는 참·거짓이어야 합니다", file=sys.stderr)
            continue
        # 큐 종류가 학습을 멈추겠다고 말할 수는 없다. 멈춰 두기는 **시한이 있는** 곁다리라야
        # 성립한다 — 깨워 줄 사람이 없으면 얼어붙은 학습이 그대로 남기 때문이다.
        if preempt and lane != "side":
            print(f"[sparkq] 종류 파일을 읽지 않습니다 {path}: preempt는 side 종류만 쓸 수 있습니다", file=sys.stderr)
            continue
        spec["preempt"] = preempt
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


def queued_jobs() -> list[dict]:
    """줄에 선 작업들. 파일 이름 순서가 곧 줄 순서다."""
    jobs = []
    for path in queued_files():
        job = read_json(path)
        if job is not None and job.get("id"):
            jobs.append(job)
    return jobs


def _rewrite_order(order: list[dict]) -> None:
    """줄 전체를 이 순서로 다시 적는다.

    이름이 곧 순서이므로(`queue_path`) 바꾸는 것은 이름뿐이고, 순서를 적은 색인 파일은
    여전히 없다. 우선순위를 0부터 다시 매기는 이유는 예전 `top`이 `lowest - 1`로 한 칸씩
    내려가다 0에서 바닥을 쳤기 때문이다 — 그 뒤로는 맨 앞으로 보내도 앞으로 가지 않았다.
    지금은 줄에 선 것들이 늘 `0..N-1`을 쓰고, 새로 걸리는 작업은 `DEFAULT_PRIORITY`(5000)라
    언제나 뒤에 선다.

    **반드시 `QUEUE_LOCK` 안에서 부른다.** 쓰기와 지우기 사이에 `tick`이 `queued_files()[0]`을
    집으면 같은 작업이 두 이름으로 잠깐 보이고, 그중 하나가 줄에 유령으로 남는다.
    """
    planned = []
    for index, job in enumerate(order):
        created_ns = int(round(float(job.get("created_at", time.time())) * 1e9))
        planned.append((queue_path(index, created_ns, job["id"]), index, job))
    keep = {path for path, _, _ in planned}
    for path, index, job in planned:
        job["priority"] = index
        write_json(path, job)
    for path in queued_files():
        if path not in keep:
            path.unlink(missing_ok=True)


def move(job_id: str, before: str | None) -> dict:
    """줄에 선 작업 하나를 다른 작업 **바로 앞**으로 옮긴다. `before`가 없으면 맨 뒤로.

    자리를 번호가 아니라 **다른 작업의 이름**으로 받는 이유가 있다. 화면이 3번을 2번으로
    옮기려는 순간에 앞의 것이 시작해 버리면 번호는 다른 자리를 가리키지만, "저 작업 앞"은
    그대로 그 자리다. 자는 동안 줄이 저절로 줄어드는 큐라서 번호는 미덥지 않다.
    """
    with QUEUE_LOCK:
        jobs = queued_jobs()
        index = next((i for i, job in enumerate(jobs) if job["id"] == job_id), None)
        if index is None:
            raise Missing(f"그런 작업이 없습니다: {job_id}")
        if before is not None and before == job_id:
            raise Invalid("자기 앞으로는 옮길 수 없습니다")
        moving = jobs.pop(index)
        if before is None:
            jobs.append(moving)
        else:
            target = next((i for i, job in enumerate(jobs) if job["id"] == before), None)
            if target is None:
                raise Missing(f"그런 작업이 없습니다: {before}")
            jobs.insert(target, moving)
        _rewrite_order(jobs)
        return moving


def move_to_top(job_id: str) -> dict:
    """맨 앞으로. `move`와 같은 자리를 쓰므로 우선순위가 바닥을 치는 일이 없다."""
    with QUEUE_LOCK:
        jobs = queued_jobs()
        index = next((i for i, job in enumerate(jobs) if job["id"] == job_id), None)
        if index is None:
            raise Missing(f"그런 작업이 없습니다: {job_id}")
        moving = jobs.pop(index)
        jobs.insert(0, moving)
        _rewrite_order(jobs)
        return moving


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


# ---------------------------------------------------------------- 누구의 GPU 프로세스인가

def container_of(pid: str) -> str | None:
    """이 프로세스가 들어 있는 도커 컨테이너의 id. 컨테이너 밖이면 `None`.

    **세션 트리만으로는 부족하기 때문에 있는 함수다.** `docker exec`로 띄운 학습의 GPU
    프로세스는 tmux pane의 자손이 아니다. 부모를 거슬러 올라가면 pane이 아니라
    containerd-shim이 나온다(2026-09-07 실측: Isaac 학습의 GPU pid 2548722의 조상은
    2548701 → 288832 containerd-shim → 1). 그래서 큐는 **자기가 띄운 학습의 GPU
    프로세스를 남의 것으로 세고 있었다.** 컨테이너 id는 그 프로세스가 어느 문으로
    들어갔는지 말해 주는 유일한 흔적이다.
    """
    try:
        text = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
    except OSError:
        return None
    found = re.search(r"[0-9a-f]{64}", text)
    return found.group(0) if found else None


def session_containers(session: str) -> set[str]:
    """이 세션이 열어 두고 말을 걸고 있는 컨테이너들의 id.

    명령줄을 파싱하지 않고 **이름을 맞춰 본다.** `docker exec [-e A=1] 이름 …`의 플래그를
    해석하려 들면 종류 파일이 옵션을 하나 더 쓸 때마다 틀리기 시작한다. 도는 컨테이너의
    이름은 도커에게 물으면 되고, 그 이름이 세션의 명령줄에 토큰으로 그대로 있으면 이
    세션이 그 컨테이너를 쓰는 것이다.
    """
    pids = session_process_ids(session)
    if not pids:
        return set()
    words: set[str] = set()
    for pid in pids:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        words.update(raw.decode("utf-8", errors="replace").split("\0"))
    if not words:
        return set()
    try:
        out = subprocess.run(
            ["docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if out.returncode != 0:
        return set()
    found = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in words:
            found.add(parts[0])
    return found


def job_gpu_pids(job: dict, apps: list[dict] | None = None) -> list[str]:
    """이 작업의 것이라고 **말할 수 있는** GPU 프로세스들.

    두 갈래로 본다. 세션의 자손이면 그대로 이 작업의 것이고(가상환경에서 도는 학습),
    자손이 아니면 그 프로세스가 있는 컨테이너를 이 작업이 열어 두었는지를 본다
    (`docker exec`로 도는 학습).

    둘 다 아니면 이 작업의 것이라고 **말하지 않는다.** 모르는 것을 내 것으로 세는 쪽이
    훨씬 위험하다 — 그 판단 위에서 프로세스를 멈추기 때문이다.
    """
    session = job.get("session", "")
    if apps is None:
        apps = compute_apps() or []
    tree = session_process_ids(session) or set()
    mine = [str(app.get("pid")) for app in apps if str(app.get("pid")) in tree]
    rest = [str(app.get("pid")) for app in apps if str(app.get("pid")) not in tree]
    if rest:
        containers = session_containers(session)
        if containers:
            mine.extend(pid for pid in rest if container_of(pid) in containers)
    return mine


def foreign_apps(apps: list[dict], job: dict | None, side: dict | None) -> list[dict]:
    """큐가 자기 것이라고 말할 수 없는 GPU 프로세스만 남긴다.

    **이 값은 경고문이 되어 사람에게 간다.** 자기 학습을 남의 것으로 부르면 사람은 있지도
    않은 사고를 보게 되고, 몇 번 그러고 나면 그 경고를 안 읽게 된다. 그래서 여기서는
    컨테이너까지 보고 판단한다.

    문지기(`_tick`)는 이 함수를 쓰지 않는다. 거기서 묻는 것은 "이것이 남의 것인가"가
    아니라 "GPU가 비었는가"이고, 그 질문의 안전한 답은 언제나 **모르면 기다린다**이다.
    """
    mine: set[str] = set()
    for owner in (job, side):
        if owner is not None:
            mine.update(job_gpu_pids(owner, apps))
    return [app for app in apps if str(app.get("pid")) not in mine]


# ---------------------------------------------------------------- 멈춰 두기(선점)

def signal_pids(pids: list[str], number: int) -> list[str]:
    """보낼 수 있는 것에만 보내고, 실제로 보낸 것을 돌려준다.

    이미 사라진 프로세스는 조용히 건너뛴다. 얼리는 쪽에서는 "몇 개를 실제로 얼렸나"가
    판단에 필요하고, 깨우는 쪽에서는 하나가 없어도 나머지를 마저 깨워야 한다.
    """
    sent = []
    for pid in pids:
        try:
            os.kill(int(pid), number)
        except (OSError, ValueError):
            continue
        sent.append(str(pid))
    return sent


def preempt_targets(job: dict) -> list[str]:
    """이 학습에서 **멈춰야 하는** 프로세스들.

    GPU를 볼 수 있는 기계에서는 GPU를 쥔 프로세스만 멈춘다. 적게 건드리는 쪽이 안전하고,
    비켜 줘야 하는 것은 GPU 연산이지 셸이 아니다. 호스트에 남아 대기하는 `docker exec`
    클라이언트는 연산을 쓰지 않으므로 그대로 둔다.

    GPU를 볼 수 없는 기계(맥)에는 그 목록이 아예 없다. 거기서는 세션 트리 전체를 멈춘다 —
    그 기계의 학습은 컨테이너 밖에서 돌아 트리가 곧 전부다.
    """
    apps = compute_apps()
    if apps is None:
        return sorted(session_process_ids(job.get("session", "")) or [], key=int)
    return sorted(job_gpu_pids(job, apps), key=int)


def preempt_for(side: dict) -> dict | None:
    """곁다리를 위해 도는 학습을 얼린다. 얼릴 것이 없으면 `None`.

    `SIGSTOP`을 받은 프로세스는 GPU에 새 커널을 내보내지 않는다. 이미 올라간 것이 끝나면
    연산은 곧 0이 되고, 통합메모리라 **자리는 그대로 둔 채** 연산만 비켜 준다.

    되돌릴 수 없는 일을 하지 않는 것이 이 함수의 성격이다. 죽이지 않으므로 스텝도
    옵티마이저 상태도 잃지 않는다.
    """
    training = current_job()
    if training is None or not session_alive(training.get("session", "")):
        return None
    targets = preempt_targets(training)
    if not targets:
        # 얼릴 대상을 모르는 채로 옆에서 추론을 시작하면 둘이 GPU를 나눠 쓴다. 그것은
        # 이 기능이 막으려던 바로 그 상태이므로, 모르면 시작하지 않는다.
        raise Conflict(
            "도는 학습의 GPU 프로세스를 찾지 못해 멈출 수 없습니다. 학습이 이제 막 "
            "시작하는 중이면 CUDA가 올라온 뒤(수십 초) 다시 눌러 주세요."
        )
    stopped = signal_pids(targets, signal.SIGSTOP)
    if not stopped:
        raise Conflict("학습 프로세스에 멈춤 신호를 보내지 못했습니다: " + ", ".join(targets))
    record = {
        "job": training["id"],
        "title": training.get("title"),
        "session": training.get("session"),
        "pids": stopped,
        "at": time.time(),
        "side": side.get("id"),
        "side_kind": side.get("kind"),
    }
    write_json(PREEMPTED_FILE, record)
    return record


def resume_preempted() -> dict | None:
    """얼려 둔 학습을 깨운다. 없으면 `None`.

    **여러 번 불러도 안전하다.** 도는 프로세스에 `SIGCONT`는 아무 일도 아니고, PID가
    재사용됐더라도 CONT는 해를 끼치지 않는다(위험한 방향은 얼리는 쪽이고, 그쪽은 부를
    때마다 대상을 새로 고른다). 그래서 깨우는 자리를 여럿 두었다 — 곁다리가 끝날 때,
    데몬이 올라올 때, 박자마다. 하나라도 살아 있으면 학습은 깨어난다.
    """
    record = read_json(PREEMPTED_FILE)
    if record is None:
        return None
    signal_pids([str(pid) for pid in (record.get("pids") or [])], signal.SIGCONT)
    paused = max(0.0, time.time() - float(record.get("at") or time.time()))
    job = read_json(run_dir(str(record.get("job"))) / "job.json")
    if job is not None and job.get("id"):
        # 멈춰 있던 시간을 학습 기록에 더해 둔다. 걸린 시간을 그대로 두면 "5시간 걸렸다"가
        # 실제로는 4시간 학습 + 1시간 멈춤이 되고, 다음에 그 숫자로 계획을 세울 수 없다.
        job["paused_seconds"] = float(job.get("paused_seconds") or 0.0) + paused
        write_json(run_dir(job["id"]) / "job.json", job)
    PREEMPTED_FILE.unlink(missing_ok=True)
    return {**record, "resumed_at": time.time(), "paused_seconds": paused}


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
#: 반복 블록 안의 `이름: 값` 한 줄. 값 뒤의 `s`(초)는 버린다.
#:
#: rsl_rl은 반복마다 서른 줄이 넘는 지표를 찍는데 지금까지 화면은 그중 하나(평균 보상)만
#: 보고 있었다. 손실 셋도, 성공률도, 보상이 **어느 항목에서** 오는지도 로그에는 이미 있다.
#: 그것을 못 보면 보상이 왜 오르내리는지 알 수 없어 학습을 눈으로 고칠 수 없다.
_RSL_METRIC = re.compile(r"^\s*([A-Za-z][\w/ .()-]*?):\s+(-?\d+(?:\.\d+)?)s?\s*$")

#: 지표를 묶는 자리. 서른 개를 한 표에 늘어놓으면 읽을 수 없다.
#:
#: 접두사로 가른다 — rsl_rl이 이미 `Episode_Reward/…`, `Curriculum/…`처럼 묶어서 찍기
#: 때문이고, 그 묶음이 곧 사람이 함께 보고 싶어 하는 단위다.
_RSL_GROUPS: list[tuple[str, str]] = [
    ("Episode_Reward/", "보상 항목"),
    ("Episode_Termination/", "끝난 이유"),
    ("Curriculum/", "커리큘럼"),
    ("Metrics/", "지표"),
]

#: 묶이지 않는 이름들을 어디에 둘지. 없으면 `그 밖`으로 간다.
_RSL_PLAIN_GROUPS: dict[str, str] = {
    "Mean reward": "학습",
    "Mean value loss": "손실",
    "Mean surrogate loss": "손실",
    "Mean entropy loss": "손실",
    "Mean episode length": "학습",
    "Mean action std": "학습",
    "Total steps": "속도",
    "Steps per second": "속도",
    "Collection time": "속도",
    "Learning time": "속도",
    "Iteration time": "속도",
}

#: 화면에 적을 이름. 없으면 로그의 이름을 그대로 쓴다.
_RSL_LABELS: dict[str, str] = {
    "Mean reward": "평균 보상",
    "Mean value loss": "가치 손실",
    "Mean surrogate loss": "대리 손실",
    "Mean entropy loss": "엔트로피 손실",
    "Mean episode length": "에피소드 길이",
    "Mean action std": "행동 표준편차",
    "Total steps": "총 스텝",
    "Steps per second": "초당 스텝",
    "Collection time": "수집 시간(초)",
    "Learning time": "학습 시간(초)",
    "Iteration time": "반복 시간(초)",
    "Metrics/success_rate": "성공률",
}


def _rsl_group(name: str) -> str:
    for prefix, label in _RSL_GROUPS:
        if name.startswith(prefix):
            return label
    return _RSL_PLAIN_GROUPS.get(name, "그 밖")


def _rsl_label(name: str) -> str:
    if name in _RSL_LABELS:
        return _RSL_LABELS[name]
    # `Episode_Reward/lifting_object` → `lifting object`. 접두사는 묶음 이름이 이미 말한다.
    tail = name.split("/", 1)[-1]
    return tail.replace("_", " ")


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
        raise Missing(f"그런 작업이 없습니다: {job_id}")
    flavour = job.get("progress", "none")
    path = run_dir(job_id) / "run.log"
    train: list[list[float]] = []
    held_out: list[list[float]] = []
    #: rsl_rl만 채운다. 반복 블록의 나머지 지표를 이름별로 모아 둔 것.
    rsl_extra: dict[str, list[list[float]]] = {}

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
        # 반복 블록 안의 숫자를 **전부** 모은다. 이름 하나가 곡선 하나다.
        iteration = None
        collected: dict[str, list[list[float]]] = {}
        for line in log_lines(path):
            found = _RSL_ITER.search(line)
            if found is not None:
                iteration = int(found.group(1))
                continue
            if iteration is None:
                continue
            hit = _RSL_METRIC.search(line)
            if hit is None:
                continue
            name = hit.group(1).strip()
            # `Learning iteration`은 위에서 이미 읽었고, 시계 모양(`ETA`, `Time elapsed`)은
            # 이 규칙에 걸리지 않는다.
            if name in {"Learning iteration"}:
                continue
            try:
                value = float(hit.group(2))
            except ValueError:
                continue
            collected.setdefault(name, []).append([float(iteration), value])
        train = collected.pop("Mean reward", [])
        rsl_extra = collected

    out = []
    if train:
        name, label, axis = (
            ("reward", "평균 보상", "반복") if flavour == "rsl_rl" else ("loss", "학습 손실", "스텝")
        )
        group = "학습" if flavour == "rsl_rl" else "손실"
        out.append({"name": name, "label": label, "axis": axis, "group": group,
                    "points": thin(train, limit)})
    # 나머지 지표. 평균 보상 뒤에 오되 순서는 로그에 나온 순서 그대로다 — rsl_rl이 찍는
    # 차례가 곧 사람이 읽는 차례이고, 여기서 다시 정렬하면 그 뜻이 사라진다.
    for name, points in rsl_extra.items():
        if len(points) < 2:
            continue
        out.append({
            "name": name, "label": _rsl_label(name), "axis": "반복",
            "group": _rsl_group(name), "points": thin(points, limit),
        })
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
        # 이 곁다리가 딸린 학습. 종류가 `watches`로 말한 종류의 학습이 도는 중이면 그 번호를
        # 적어 두고, 그 학습이 서면 이 곁다리도 함께 선다(`stop_following_side`).
        "follows": (
            training["id"]
            if training is not None and training.get("kind") in (spec.get("watches") or [])
            else None
        ),
    }

    directory = run_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    command_script = directory / "side-command.sh"
    log = directory / "run.log"
    command_script.write_text(run_script_text(job["command"], log), encoding="utf-8")
    command_script.chmod(0o755)

    # 얼리는 것은 세션을 만들기 **전**이다. 순서가 반대이면 곁다리가 먼저 GPU를 잡고
    # 학습과 겹치는 구간이 생긴다 — 그 겹침을 없애려고 만든 기능이다.
    preempted = preempt_for(job) if spec.get("preempt") else None
    if preempted:
        job["preempted"] = preempted

    script = directory / "run.sh"
    # 최초 만료는 데몬이, 연장 가능한 절대 상한은 이 timeout도 함께 지킨다. 최초 600초로
    # 감싸면 API로 10분을 연장해도 먼저 죽으므로 wrapper에는 extendable_until의 상한을 쓴다.
    if preempted:
        # 데몬과 **별개로** 이 세션이 스스로 학습을 깨운다. 데몬이 죽어도, 기계가 이 세션만
        # 남기고 이상해져도, 이 셸이 끝나는 순간 CONT가 나간다. 얼어붙은 학습이 남는 것이
        # 이 기능의 유일한 큰 사고이므로 깨우는 손을 둘로 둔다.
        # `exec`를 쓰지 않는 이유가 이것이다 — 셸을 갈아 끼우면 trap을 실행할 셸이 없다.
        wake = " ".join(shlex.quote(str(pid)) for pid in preempted["pids"])
        script.write_text(
            "#!/usr/bin/env bash\nset -o pipefail\n"
            f"trap 'kill -CONT {wake} 2>/dev/null' EXIT INT TERM\n"
            f"{limiter} bash {shlex.quote(str(command_script))}\n",
            encoding="utf-8",
        )
    else:
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
        # 세션이 안 떴으면 깨워 줄 trap도 없다. 여기서 되돌리지 않으면 학습은 아무도
        # 시작하지 못한 곁다리 때문에 얼어붙은 채로 남는다.
        resume_preempted()
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
    # 곁다리가 어떻게 끝났든(정상·취소·만료·실패) 얼려 둔 학습은 여기서 깨어난다.
    resume_preempted()
    return job


def stop_following_side(job_id: str) -> None:
    """이 학습에 딸려 뜬 곁다리(뷰어)를 함께 세운다. 큐 락 안에서 부른다.

    학습은 섰는데 그 학습의 뷰어만 남아 있는 모양은 사람에게 고장으로 보이고(2026-09-07
    사용자 요청), 큐에는 GPU를 쥔 채 남은 프로세스다. 뷰어는 시한이 있어 언젠가 꺼지지만,
    딸린 학습이 서는 순간이 곧 그 뷰어가 볼 것이 없어지는 순간이다. 학습이 끝난 **뒤에**
    띄운 뷰어(최종 정책 보기)는 `follows`가 없으므로 그대로 둔다.
    """
    side = current_side()
    if side is None or side.get("follows") != job_id:
        return
    side["note"] = f"학습 {job_id}이 서서 함께 세웠습니다"
    write_json(run_dir(side["id"]) / "job.json", side)
    _stop_side()


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
    # 곁다리가 없는데 얼려 둔 학습이 남아 있으면 깨운다. `finalize_side`가 이미 깨우므로
    # 보통은 할 일이 없고, 표만 남는 드문 경우(밖에서 세션을 지웠다든가)의 그물이다.
    if side is None and PREEMPTED_FILE.exists():
        resume_preempted()

    job = current_job()
    if job is not None:
        if session_alive(job.get("session", "")):
            write_json(run_dir(job["id"]) / "progress.json", progress_of(job))
            return
        finalize(job)
        stop_following_side(job["id"])
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
        stop_following_side(job["id"])
    side = current_side()
    if side is not None and not session_alive(side.get("session", "")):
        finalize_side(side)
        side = None
    # 얼려 둔 학습이 있는데 그것을 얼린 곁다리가 없으면 깨운다. 데몬이 죽어 있는 동안
    # 곁다리가 끝났거나, 기계가 재부팅됐거나, 표만 남고 프로세스는 사라진 경우다.
    if side is None or not session_alive(side.get("session", "")):
        resume_preempted()


def stop_grace_of(job: dict) -> float:
    """이 작업이 스스로 정리할 시간을 얼마나 주는가.

    종류 파일이 `stop_grace_seconds`로 말한다. 상한을 두는 이유는 이 기다림이 큐의 락을
    쥐고 있기 때문이다 — 종류 하나가 중지를 영영 붙들 수는 없다.
    """
    spec = load_kinds().get(job.get("kind", "")) or {}
    try:
        value = float(spec.get("stop_grace_seconds", DEFAULT_STOP_GRACE))
    except (TypeError, ValueError):
        return DEFAULT_STOP_GRACE
    return max(0.0, min(value, MAX_STOP_GRACE))


def end_session(job: dict) -> None:
    """`C-c`를 주고, 정리가 끝나기를 **기다렸다가**, 그래도 남으면 세션을 죽인다.

    기다리는 것이 이 함수의 전부다. 전에는 2초를 자고 무조건 `kill-session`이었는데,
    그 2초가 작업의 정리 스크립트보다 짧으면 정리가 중간에 끊긴다. 실제로 그렇게 됐다:
    Isaac 뷰어의 EXIT trap은 컨테이너 안에 `pkill -INT`를 보내고 5초를 기다렸다가
    `pkill -KILL`을 보내는데, 그 `sleep 5` 도중에 세션이 죽어 **KILL이 영영 실행되지
    않았다.** 남은 `play.py`가 GPU를 쥔 채로 있으면 큐는 다음 작업을 꺼내지 못한다 —
    2026-09-06에 그렇게 일곱 시간 반이 막혔다.

    `docker exec`는 클라이언트를 죽여도 컨테이너 안의 프로세스에 신호를 보내지 않는다.
    그래서 컨테이너 안을 정리할 수 있는 것은 작업 자신의 trap뿐이고, 큐가 할 수 있는
    유일한 도움은 **그 trap이 끝날 때까지 기다리는 것**이다.

    빨리 끝나는 작업이 손해를 보지도 않는다. 세션이 사라지는 즉시 돌아오므로, 전의
    무조건 2초보다 오히려 빠르다.
    """
    session = job.get("session", "")
    if not session:
        return
    subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], capture_output=True, timeout=30)
    deadline = time.monotonic() + stop_grace_of(job)
    while time.monotonic() < deadline:
        if not session_alive(session):
            return
        time.sleep(0.5)
    if session_alive(session):
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, timeout=30)


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
        raise Missing(f"그런 작업이 없습니다: {job_id}")
    (run_dir(job_id) / "cancelled").write_text("1", encoding="utf-8")
    end_session(job)
    result = finalize(current_job() or job)
    stop_following_side(job_id)
    return result


def _finished_record(job_id: str) -> dict:
    """끝난 작업의 기록. 아직 대기 중이거나 도는 것이면 409 — 그것은 `rm`의 일이다."""
    if find_queued(job_id) is not None:
        raise Conflict(f"아직 대기 중인 작업입니다: {job_id}")
    for live in (current_job(), current_side()):
        if live is not None and live.get("id") == job_id:
            raise Conflict(f"아직 도는 작업입니다: {job_id}")
    job = read_json(run_dir(job_id) / "job.json")
    if job is None:
        raise Missing(f"그런 작업이 없습니다: {job_id}")
    return job


def rename(job_id: str, title: str) -> dict:
    """끝난 작업의 이름을 바꾼다. 기록(job.json)의 `title`만 바뀌고 로그·산출물·tmux 세션 이름은 그대로다.

    이름은 사람이 나중에 알아보려고 붙이는 것이라 자유 문자열이되, 한 줄(공백 정리)·80자 안이다.
    """
    title = " ".join(str(title).split())
    if not title:
        raise Invalid("이름이 비어 있습니다")
    if len(title) > 80:
        raise Invalid("이름은 80자 안이어야 합니다")
    with QUEUE_LOCK:
        job = _finished_record(job_id)
        job["title"] = title
        write_json(run_dir(job_id) / "job.json", job)
        return job


def forget(job_id: str) -> dict:
    """끝난 작업을 내역에서 뺀다. 기록 폴더(job.json, run.log)를 `~/.sparkq/trash/<id>`로 옮긴다.

    지우지 않고 옮기는 이유는 되돌릴 수 있어야 해서다. 학습 산출물(체크포인트)은 여기 없다 —
    그것은 실행 폴더에 있고 `rm-run`의 몫이다.
    """
    with QUEUE_LOCK:
        _finished_record(job_id)
        TRASH_DIR.mkdir(parents=True, exist_ok=True)
        target = TRASH_DIR / job_id
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(run_dir(job_id)), str(target))
        return {"id": job_id, "forgotten": True, "moved_to": str(target)}


def stop_side() -> dict:
    with QUEUE_LOCK:
        return _stop_side()


def _stop_side(*, expired: bool = False) -> dict:
    job = current_side()
    if job is None:
        raise Missing("지금 곁다리가 없습니다")
    directory = run_dir(job["id"])
    marker = directory / ("expired" if expired else "cancelled")
    marker.write_text("1", encoding="utf-8")
    end_session(job)
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
            raise Missing("지금 곁다리가 없습니다")
        expires = float(job["expires_at"]) + amount
        if expires > float(job["extendable_until"]):
            raise Invalid("곁다리는 시작 뒤 3600초를 넘겨 연장할 수 없습니다")
        job["expires_at"] = expires
        write_json(run_dir(job["id"]) / "job.json", job)
        return side_view(job)


def stream_ready(job: dict) -> bool | None:
    """이 곁다리가 **볼 수 있는 상태**가 됐는가. 종류가 그 표시를 말하지 않으면 `None`.

    포트가 열린 것을 준비로 삼으면 안 되기 때문에 있는 함수다. Isaac의 스트림 확장은
    시작 20초쯤에 포트를 열지만, 씬을 짓고 정책을 읽는 것은 그 뒤다. 체크포인트가 지금
    환경과 맞지 않으면 38초쯤에 죽는데, 그 사이에 사람이 붙으면 검은 화면만 본다.
    실제로 그렇게 됐고, 그래서 준비의 기준을 로그의 한 줄로 옮겼다.

    로그 꼬리가 아니라 처음부터 찾는 이유: Isaac의 시작 로그는 수백 줄이라 준비 줄이
    금세 꼬리 밖으로 밀려난다. 준비됐다가 다시 안 준비될 일은 없으므로 한 번 나오면 참이다.
    """
    spec = load_kinds().get(job.get("kind", ""))
    marker = ((spec or {}).get("stream") or {}).get("ready")
    if not marker:
        return None
    path = run_dir(job.get("id", "")) / "run.log"
    for line in log_lines(path):
        if marker in line:
            return True
    return False


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
        "live": session_alive(job["session"]),
        "started_at": job.get("started_at"),
        "expires_at": job.get("expires_at"),
        "extendable_until": job.get("extendable_until"),
        "baseline_step_seconds": job.get("baseline_step_seconds"),
        "step_seconds": step_seconds,
        # `None`은 이 종류가 준비 표시를 말하지 않는다는 뜻이고, 거짓과 다르다.
        "stream_ready": stream_ready(job),
        # 이 곁다리가 학습을 멈춰 두고 있는가. 앱은 이것이 있으면 "느려짐 31%" 대신
        # "멈춰 둠 12분"을 보여 준다 — 겹쳐 도는 것과 얼려 둔 것은 다른 이야기다.
        "preempted_job": (job.get("preempted") or {}).get("job"),
        "preempted_at": (job.get("preempted") or {}).get("at"),
    }


#: `/api/history` 한 번에 줄 수 있는 최대. 한 페이지가 커지면 이 문도 폴링만큼 무거워진다.
HISTORY_PAGE_MAX = 200


def finished_jobs() -> list[dict]:
    """끝난 것 전부, 새것부터.

    `recent`(폴링이 싣는 앞머리)와 `history`(사람이 뒤로 넘길 때)가 **같은 목록**을 본다.
    같은 판단(무엇이 끝난 것인가, 곁다리를 어떻게 셀 것인가)을 두 곳에 적으면 한쪽만
    고쳐지는 날이 오고, 그때 두 문이 서로 다른 과거를 말하게 된다.
    """
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
    return jobs


def recent(limit: int = 20) -> list[dict]:
    """최근에 끝난 것들. 새것부터."""
    return finished_jobs()[:limit]


def history(limit: int = 40, before: float | None = None) -> dict:
    """끝난 것들을 뒤로 넘겨 가며 읽는다.

    `snapshot()`이 싣는 것은 앞머리 20개뿐이다. 그 수를 늘려 해결하고 싶어지지만, 스냅숏은
    앱이 몇 초마다 다시 읽는 것이라 **폴링 한 번의 무게가 기록의 개수만큼 영영 자란다.**
    그래서 옛것은 사람이 `더 보기`를 누를 때만 이 문으로 따로 읽는다.

    `before`는 커서다 — 그 시각보다 **먼저 끝난 것**부터 준다. 페이지 번호 대신 커서를 쓰는
    이유는, 사람이 뒤를 읽는 동안에도 앞에서 새 작업이 끝나기 때문이다. 번호로 세면 그때
    목록이 한 칸씩 밀려 같은 줄을 두 번 보거나 한 줄을 건너뛴다.
    """
    limit = max(1, min(int(limit), HISTORY_PAGE_MAX))
    jobs = finished_jobs()
    total = len(jobs)
    if before is not None:
        jobs = [job for job in jobs if (job.get("finished_at") or 0) < before]
    page = jobs[:limit]
    return {
        "recent": page,
        # 이 뒤로 더 있는가. 앱이 `더 보기`를 계속 내줄지 정하는 값이다.
        "more": len(jobs) > len(page),
        # 끝난 것이 통틀어 몇 개인가. 앱이 "20개 가운데"가 아니라 "78개 가운데"라고 말한다.
        "total": total,
    }


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
        # 끝난 것이 통틀어 몇 개인가. 앱은 이 수와 위 목록의 길이를 견주어, 뒤에 더 있을
        # 때만 `과거 더 읽기`를 내준다 — 없는 과거를 부르는 단추를 만들지 않는다.
        "recent_total": len(finished_jobs()),
        "paused": PAUSED_FILE.exists(),
        "preempted": read_json(PREEMPTED_FILE),
        "foreign_sessions": train_sessions(job.get("session") if job is not None else None),
        "capabilities": capabilities(),
    }
    # 볼 수 없는 기계에서는 이 칸을 **아예 싣지 않는다.** 빈 목록으로 실으면 "확인했고
    # 비어 있다"가 되는데, 실제로는 확인할 방법이 없었던 것이다.
    if WATCHES_GPU_PROCESSES:
        # 세션의 프로세스 트리를 못 읽었을 때는 안전한 쪽으로 전체 GPU 목록을 그대로 싣는다.
        out["gpu_apps"] = foreign_apps(compute_apps() or [], job, side)
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


def directory_bytes(directory: Path) -> int:
    """폴더 하나가 차지한 바이트. 읽지 못하는 파일은 0으로 센다.

    크기를 못 읽었다고 폴더 전체를 못 세겠다고 하면, 화면은 지울 수 있는 것을 지울 수
    없는 것처럼 보여 준다. 한 파일이 빠진 합계가 합계가 없는 것보다 낫다.
    """
    total = 0
    for base, _dirs, files in os.walk(directory):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def checkpoints_of(directory: Path) -> list[dict]:
    """실행 하나가 남긴 체크포인트들.

    `pretrained_model`과 `training_state`를 **따로** 센다. 뒤의 것은 optimizer 상태이고
    이어붙일 때만 쓰인다 — 추론에는 필요 없는데 체크포인트 하나의 3분의 1쯤을 차지한다
    (실측으로 SmolVLA 2만 스텝에서 865MB 대 394MB). 두 숫자를 하나로 합쳐 놓으면 화면이
    "무엇을 지우면 무엇을 잃는가"를 말할 수 없고, 그러면 지우는 것이 늘 전부 아니면
    전무가 된다.

    `checkpoints/last`는 마지막 체크포인트를 가리키는 심볼릭 링크다. 따라가면 같은 것이
    목록에 두 번 나오고, 화면은 있지도 않은 체크포인트를 하나 더 센다.
    """
    root = directory / "checkpoints"
    out: list[dict] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return out
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir():
            continue
        model = entry / "pretrained_model"
        if not model.is_dir():
            continue
        model_bytes = directory_bytes(model)
        state = entry / "training_state"
        state_bytes = directory_bytes(state) if state.is_dir() else 0
        try:
            finished_at = model.stat().st_mtime
        except OSError:
            finished_at = 0.0
        out.append({
            "step": entry.name,
            "model_bytes": model_bytes,
            "state_bytes": state_bytes,
            "bytes": model_bytes + state_bytes,
            "finished_at": finished_at,
        })
    return out


def runs() -> list[dict]:
    """학습이 남긴 것들. `~/outputs/*` 가운데 체크포인트가 있는 실행.

    이 목록은 두 화면이 나눠 쓴다. `이어서 학습`은 **무엇을 이어붙일 수 있는가**를 묻고,
    `학습된 정책`은 **무엇이 디스크를 차지하고 있고 무엇을 팔로 보낼 수 있는가**를 묻는다.
    질문이 둘이라고 목록을 둘로 나누지 않는 이유는 사실이 하나이기 때문이다 — 같은 폴더를
    두 곳에서 세면 한쪽만 갱신되는 날이 온다.

    이어붙이기가 보는 것은 `checkpoints/last`다. 그것은 마지막 체크포인트를 가리키는
    심볼릭 링크이고, 그 이름이 곧 지금까지 간 스텝이다. 나머지(목표 스텝·정책·데이터셋)는
    그 안의 `train_config.json`에 있다 — 이어붙일 때 lerobot이 읽는 파일과 같은 것이라
    화면과 실행이 어긋나지 않는다.
    """
    out = []
    if not OUTPUT_ROOT.is_dir():
        return out
    # 도는 작업과 tmux 세션은 실행마다 다시 묻지 않는다. 이 목록은 화면이 3초마다 읽고,
    # 실행 하나당 tmux를 한 번씩 부르면 그 값이 곧 폴링 비용이 된다.
    claims = job_claims()
    sessions = train_sessions()
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
        checkpoints = checkpoints_of(directory)
        out.append({
            "name": directory.name,
            "step": step,
            "steps": meta.get("steps") or 0,
            "policy": (meta.get("policy") or {}).get("type") or "",
            "dataset": (meta.get("dataset") or {}).get("repo_id") or "",
            "updated_at": updated,
            "checkpoints": checkpoints,
            # 옆자리(`.runs/<run>`)는 로그와 메타뿐이라 크기에 넣지 않는다. 지울 때는
            # 함께 지우지만, 화면이 "5.0GB를 되찾는다"고 말할 때의 근거는 체크포인트다.
            "bytes": sum(item["bytes"] for item in checkpoints),
            # 이 실행을 지금 누가 쓰고 있는가. 지울 수 있는지를 화면이 미리 알아야
            # 버튼을 눌러 보고 나서 409를 보는 일이 없다.
            "in_use": run_in_use(directory.name, claims=claims, sessions=sessions),
        })
    # 최근에 손댄 것이 위로. 밤마다 이어 붙이는 것은 거의 늘 어젯밤 것이다.
    out.sort(key=lambda item: item["updated_at"], reverse=True)
    return out


# ---------------------------------------------------------------- 산출물 지우기

def run_directory(run: str) -> Path:
    """`~/outputs/<run>`. 이름 검사만으로는 모자란 자리다.

    `NAME`을 통과한 이름이라도 그 폴더가 심볼릭 링크로 바깥을 가리킬 수 있다. 지우는
    경로에서는 이름이 아니라 **실제로 닿는 자리**를 봐야 한다.
    """
    if not NAME.match(run):
        raise Invalid("실행 이름 형식이 아닙니다")
    directory = OUTPUT_ROOT / run
    if not directory.is_dir():
        raise Missing(f"그런 학습이 없습니다: {run}")
    try:
        resolved = directory.resolve()
        root = OUTPUT_ROOT.resolve()
    except OSError as error:
        raise Invalid(f"경로를 확인하지 못했습니다: {error}") from None
    if root not in resolved.parents:
        raise Invalid("실행 폴더가 outputs 밖을 가리킵니다")
    return directory


def job_claims() -> dict[str, str]:
    """큐가 아는 작업들이 가리키는 실행. 실행 이름 → 사람이 읽을 한 줄.

    실행 이름을 세션에서 되읽는다. 계약이 `session: train-${run}`을 강제하므로 이것이
    종류를 가리지 않는 유일한 길이다 — `params`에 `run`이 들어 있는 종류는
    `lerobot-resume`뿐이고, `lerobot-train`은 그 이름을 걸 때 만들어 명령에만 적는다.
    """
    out: dict[str, str] = {}

    def claim(job: dict | None, sentence: str) -> None:
        if not job:
            return
        session = job.get("session") or ""
        name = session.removeprefix("train-") if session.startswith("train-") else ""
        name = name or (job.get("params") or {}).get("run") or ""
        if name and name not in out:
            out[name] = sentence.format(id=job.get("id"))

    claim(current_job(), "{id} 작업이 지금 이 실행에 쓰고 있습니다")
    for path in queued_files():
        claim(read_json(path), "대기 중인 {id} 작업이 이 실행을 가리킵니다")
    return out


def run_in_use(
    run: str, *, claims: dict[str, str] | None = None, sessions: list[str] | None = None
) -> str | None:
    """이 실행을 쓰고 있는 것. 없으면 `None`.

    도는 학습이 쓰고 있는 폴더를 지우면 그 학습은 곧바로 죽지 않는다. 다음 체크포인트를
    쓸 때가 되어서야 죽고, 그것은 몇 시간 뒤다 — 아침에 남는 것은 실패 한 줄과 날아간
    밤 하나다.

    **아직 시작하지 않은 것까지 본다.** 대기 중인 `lerobot-resume`이 가리키는 실행을
    지우면 그 작업은 새벽에 시작해 몇 초 만에 죽는다. 걸 때 이름을 검사해 둔 것이 그때는
    이미 참이 아니게 된다.

    `claims`와 `sessions`는 목록을 그릴 때 한 번만 읽으려고 받는다. 지우는 자리에서는
    넘기지 않는다 — 그 한 번은 값이 가장 최근이어야 하고, 실행 하나에 tmux를 한 번 더
    묻는 값은 지우는 일에 비하면 없는 것과 같다.
    """
    if claims is None:
        claims = job_claims()
    reason = claims.get(run)
    if reason:
        return reason
    # 큐 밖에서 손으로 띄운 학습도 같은 이름 규칙을 쓴다. 큐가 모르는 학습이라고 해서
    # 그 산출물을 지워도 되는 것은 아니다.
    session = f"train-{run}"
    alive = session in sessions if sessions is not None else session_alive(session)
    return f"tmux 세션 `{session}`이 살아 있습니다" if alive else None


def repoint_last(directory: Path) -> None:
    """`checkpoints/last`가 없어진 곳을 가리키면 남은 것 가운데 마지막으로 옮긴다.

    이 한 줄이 없으면 체크포인트 하나를 지운 대가로 **실행 전체가 목록에서 사라진다.**
    `runs()`가 실행을 알아보는 표지가 `last/pretrained_model/train_config.json`이고,
    끊어진 링크는 그 파일이 없는 것과 구별되지 않기 때문이다. 지우려던 것은 체크포인트
    하나였는데 결과가 "그런 학습은 없습니다"가 된다.
    """
    root = directory / "checkpoints"
    link = root / "last"
    if not link.is_symlink():
        return
    if (link / "pretrained_model").is_dir():
        return
    remaining = [item["step"] for item in checkpoints_of(directory)]
    link.unlink(missing_ok=True)
    if remaining:
        link.symlink_to(sorted(remaining)[-1])


def delete_run(run: str) -> dict:
    """실행 하나를 통째로. 옆자리(`.runs/<run>`의 로그와 메타)도 함께 간다."""
    directory = run_directory(run)
    blocker = run_in_use(run)
    if blocker:
        raise Conflict(f"지울 수 없습니다 — {blocker}")
    freed = directory_bytes(directory)
    side = OUTPUT_ROOT / RUN_SIDE_DIR / run
    if side.is_dir():
        freed += directory_bytes(side)
        shutil.rmtree(side, ignore_errors=True)
    shutil.rmtree(directory)
    return {"run": run, "freed_bytes": freed}


def delete_checkpoint(run: str, step: str, *, only_state: bool = False) -> dict:
    """체크포인트 하나, 또는 그 안의 `training_state`만.

    `training_state`만 지우는 길을 따로 둔 이유는 두 가지를 잃는 것이 다르기 때문이다.
    그것만 지우면 **이어붙일 권리**를 버리고 가중치는 남으므로, 그 체크포인트는 여전히
    팔로 보내 돌릴 수 있다. 통째로 지우면 둘 다 사라진다.
    """
    directory = run_directory(run)
    if not NAME.match(step):
        raise Invalid("체크포인트 이름 형식이 아닙니다")
    target = directory / "checkpoints" / step
    if target.is_symlink() or not target.is_dir():
        raise Missing(f"그런 체크포인트가 없습니다: {run}/{step}")
    blocker = run_in_use(run)
    if blocker:
        raise Conflict(f"지울 수 없습니다 — {blocker}")
    if only_state:
        state = target / "training_state"
        if not state.is_dir():
            raise Missing(f"이 체크포인트에는 이미 optimizer 상태가 없습니다: {run}/{step}")
        freed = directory_bytes(state)
        shutil.rmtree(state)
        return {"run": run, "step": step, "freed_bytes": freed, "resumable": False}
    freed = directory_bytes(target)
    shutil.rmtree(target)
    repoint_last(directory)
    return {"run": run, "step": step, "freed_bytes": freed}


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

def _query_number(query: dict, name: str) -> float | None:
    """쿼리에서 숫자 하나. 없으면 `None`, 숫자가 아니면 400으로 거절한다."""
    values = query.get(name) or []
    if not values:
        return None
    try:
        return float(values[0])
    except (TypeError, ValueError):
        raise Invalid(f"{name}는 숫자여야 합니다: {values[0]!r}") from None


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
        # 산출물을 지우는 길. 큐가 자기 기계의 디스크를 소유하므로, 팔이 붙은 서버가
        # 이 기계에 ssh로 들어와 지우는 구조를 만들지 않는다.
        if method == "DELETE" and len(parts) >= 3 and parts[:2] == ["api", "runs"]:
            run = parts[2]
            if len(parts) == 3:
                return delete_run(run)
            if len(parts) == 5 and parts[3] == "checkpoints":
                return delete_checkpoint(run, parts[4])
            if len(parts) == 6 and parts[3] == "checkpoints" and parts[5] == "training_state":
                return delete_checkpoint(run, parts[4], only_state=True)
        if method == "GET" and parts == ["api", "queue"]:
            return snapshot()
        # 끝난 것의 과거. 스냅숏은 앞머리만 싣고, 뒤는 사람이 요청할 때만 이 문으로 나간다.
        if method == "GET" and parts == ["api", "history"]:
            query = urllib.parse.parse_qs(self.path.partition("?")[2])
            limit = _query_number(query, "limit")
            return history(
                limit=40 if limit is None else int(limit),
                before=_query_number(query, "before"),
            )
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
            if method == "POST" and parts[3:] == ["move"]:
                payload = self._body()
                before = payload.get("before")
                if before is not None:
                    before = str(before)
                    if not NAME.match(before):
                        raise Invalid("작업 번호 형식이 아닙니다")
                return move(job_id, before)
            if method == "POST" and parts[3:] == ["title"]:
                return rename(job_id, str(self._body().get("title", "")))
            if method == "DELETE" and parts[3:] == ["record"]:
                return forget(job_id)
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
            self._send({"detail": str(error)}, 404)
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


def human_bytes(size: float | None) -> str:
    """디스크 크기를 사람이 읽는 한 마디로. 1024로 나눈다 — `du -h`와 같은 값이어야
    터미널에서 본 것과 화면에서 본 것이 어긋나지 않는다."""
    if not size:
        return "0B"
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def show(snap: dict) -> None:
    running = snap.get("running")
    if running:
        detail = running.get("progress_detail") or {}
        step, steps = detail.get("step"), detail.get("steps")
        bar = f"{step}/{steps}" if step and steps else "시작하는 중"
        eta = human(detail.get("eta_seconds"))
        print(f"▶ {running['id']}  {running['title']}  {bar}  남은 시간 {eta}")
        frozen = snap.get("preempted")
        if frozen:
            held = human(time.time() - float(frozen.get("at") or time.time()))
            print(f"   ⏸ 곁다리({frozen.get('side_kind')}) 때문에 멈춰 두었습니다 — {held} 째. 진행은 그대로입니다.")
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
    move_cmd = sub.add_parser("mv", help="줄에 선 작업을 다른 작업 앞으로 (뒤 인자가 없으면 맨 뒤로)")
    move_cmd.add_argument("id")
    move_cmd.add_argument("before", nargs="?", help="이 작업 바로 앞에 세운다")
    rename_cmd = sub.add_parser("rename", help="끝난 작업의 이름을 바꾼다")
    rename_cmd.add_argument("id")
    rename_cmd.add_argument("title", nargs="+", help="새 이름 (여러 단어면 공백으로 이어진다)")
    forget_cmd = sub.add_parser("forget", help="끝난 작업을 내역에서 뺀다 (기록은 ~/.sparkq/trash 로)")
    forget_cmd.add_argument("id")
    top.add_argument("id")
    log = sub.add_parser("log", help="로그 꼬리")
    log.add_argument("id")
    sub.add_parser("pause", help="다음 작업을 꺼내지 않는다")
    sub.add_parser("resume", help="다시 꺼낸다")
    sub.add_parser("runs", help="학습이 남긴 것과 그 크기")
    remove_run = sub.add_parser("rm-run", help="학습이 남긴 것을 통째로 지운다")
    remove_run.add_argument("run")
    remove_ckpt = sub.add_parser("rm-ckpt", help="체크포인트 하나를 지운다")
    remove_ckpt.add_argument("run")
    remove_ckpt.add_argument("step")
    remove_ckpt.add_argument(
        "--state-only", action="store_true",
        help="optimizer 상태만 지운다 — 가중치는 남으므로 팔에서는 계속 쓸 수 있다",
    )
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
    elif args.command == "mv":
        job = call("POST", f"/api/queue/{args.id}/move", {"before": args.before})
        print(f"옮겼습니다: {job['id']}" + (f"  → {args.before} 앞" if args.before else "  → 맨 뒤"))
    elif args.command == "rename":
        job = call("POST", f"/api/queue/{args.id}/title", {"title": " ".join(args.title)})
        print(f"{job['id']}: {job['title']}")
    elif args.command == "forget":
        result = call("DELETE", f"/api/queue/{args.id}/record")
        print(f"내역에서 뺐습니다: {result['id']}  → {result['moved_to']}")
    elif args.command == "log":
        for line in call("GET", f"/api/queue/{args.id}/log")["lines"]:
            print(line)
    elif args.command in {"pause", "resume"}:
        state = call("POST", "/api/queue/pause", {"paused": args.command == "pause"})
        print("일시정지" if state["paused"] else "재개")
    elif args.command == "runs":
        for run in call("GET", "/api/runs")["runs"]:
            head = f"{run['name']}  {run['policy']}  {run['step']}스텝  {human_bytes(run['bytes'])}"
            print(f"{head}  ← {run['in_use']}" if run.get("in_use") else head)
            for checkpoint in run.get("checkpoints", []):
                state = (
                    f" + optimizer {human_bytes(checkpoint['state_bytes'])}"
                    if checkpoint["state_bytes"] else "  (이어붙일 수 없음)"
                )
                print(f"   {checkpoint['step']}  {human_bytes(checkpoint['model_bytes'])}{state}")
    elif args.command == "rm-run":
        freed = call("DELETE", f"/api/runs/{args.run}")["freed_bytes"]
        print(f"지웠습니다: {args.run}  {human_bytes(freed)} 되찾음")
    elif args.command == "rm-ckpt":
        path = f"/api/runs/{args.run}/checkpoints/{args.step}"
        freed = call("DELETE", path + ("/training_state" if args.state_only else ""))["freed_bytes"]
        what = "optimizer 상태" if args.state_only else "체크포인트"
        print(f"지웠습니다: {args.run}/{args.step}의 {what}  {human_bytes(freed)} 되찾음")


if __name__ == "__main__":
    main()
