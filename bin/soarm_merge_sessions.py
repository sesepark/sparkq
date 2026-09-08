#!/usr/bin/env python3
"""여러 수집 세션을 하나의 학습용 데이터셋으로 합친다 — 홀드아웃을 세션마다 고르게 두고.

왜 스크립트인가
---------------
lerobot은 과제마다 **마지막** ceil(n_episodes x eval_split) 개를 검증으로 뺀다
(`datasets/factory.py`의 `make_train_eval_datasets`). 그래서 여러 세션을 합친
데이터셋에서는 **합치는 순서가 곧 분할 정책**이다. 2026-09-07에 밤 57회 뒤에 낮 47회를
그대로 이어 붙였더니 검증 11개가 전부 낮이 됐다 — 조명 차이에 견디는지 보려고 합쳤는데
그것을 재는 자리에 밤이 한 개도 없었다.

이 스크립트는 그 조립을 사람 손에서 가져온다.

1. 세션마다 홀드아웃을 **세션 안에서 고르게** 뽑는다(끝 k개가 아니라). 끝만 빼면 그
   시간대의 특성(피로, 조명 변화, 손에 익은 정도)이 검증에 통째로 들어간다.
2. 학습 부분을 세션 순서대로 먼저 잇고, **검증 부분을 전부 맨 뒤에 모은다.** 그러면
   lerobot이 가져가는 "마지막 N개"가 정확히 우리가 고른 홀드아웃이 된다.
3. 구성을 `SESSIONS.json`에 남긴다. sparkq의 `kinds/lerobot-train.json`이 학습 전에 이
   파일을 읽어 검증이 비어 있는 세션이 있으면 학습을 시작하지 않는다.
4. **정책이 실제로 먹는 열만 남긴다** — `observation.state`, `observation.images.*`, `action`.
   수집기는 서보가 내주는 값을 모두 별도 열로 저장한다(부하, 속도, 온도, 전압, 상태, 전류,
   벽시계 시각 …). 그것은 원본 세션 데이터셋에 그대로 남고 분석의 몫이다. 그런데 사전학습
   config가 입력을 덮어쓰지 않는 학습 경로(`--policy.type=groot`, `--policy.type=act`)에서는
   lerobot이 데이터셋에서 특징을 읽으므로 그 열들이 전부 정책의 입력으로 선언된다. 그러면
   **추론할 때 콘솔이 `observation.wall_time`까지 만들어 줘야 하고**, 못 주면 롤아웃이 그
   자리에서 죽는다. `--keep-all-columns`로 끌 수 있다.

쓰는 법
-------
    python3 soarm_merge_sessions.py \
      --out soarm101_cube134_dnv_strat \
      --session night=soarm101_20260905_164743:5,14,24,33,43,52 \
      --session day=soarm101_20260906_052935:5,14,23,33,42 \
      --session varied=soarm101_20260907_142038

`--session 이름=데이터셋` 만 주면 홀드아웃 개수는 세션 크기에 비례해 나누고(최대잔여법),
자리는 세션 안에 고르게 흩는다. 콜론 뒤에 인덱스를 직접 적으면 그것을 쓴다 — 앞선 학습과
같은 홀드아웃을 유지해 비교 가능성을 지키고 싶을 때를 위한 것이다.

GPU를 쓰지 않는다. 영상 재인코딩은 CPU(SVT-AV1)에서 돈다.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

DEFAULT_ROOT = Path.home() / "data" / "soarm"


def parse_session(spec: str) -> tuple[str, str, list[int] | None]:
    """`이름=데이터셋` 또는 `이름=데이터셋:0,5,9` 를 뜯는다."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--session 은 이름=데이터셋 꼴이어야 합니다: {spec!r}")
    tag, rest = spec.split("=", 1)
    tag = tag.strip()
    if not tag:
        raise argparse.ArgumentTypeError(f"세션 이름이 비어 있습니다: {spec!r}")
    if ":" in rest:
        name, idx_text = rest.split(":", 1)
        indices = [int(x) for x in idx_text.split(",") if x.strip() != ""]
        if len(set(indices)) != len(indices):
            raise argparse.ArgumentTypeError(f"{tag}: 검증 인덱스가 겹칩니다 — {indices}")
        return tag, name.strip(), sorted(indices)
    return tag, rest.strip(), None


def spread(n: int, k: int) -> list[int]:
    """n개 중 k개를 세션 안에 고르게 흩어 고른다.

    구간을 k등분하고 각 구간의 가운데를 집는다. 끝 k개를 빼는 것과 달리 수집 중의
    변화(조명, 피로, 손에 익는 정도)가 검증에도 학습에도 고르게 들어간다.
    """
    if k <= 0:
        return []
    if k > n:
        raise ValueError(f"{n}회에서 {k}회를 뽑을 수 없습니다")
    step = n / k
    picked: list[int] = []
    for i in range(k):
        idx = min(n - 1, int(round((i + 0.5) * step)))
        while idx in picked:  # 반올림이 겹치면 뒤로 한 칸
            idx += 1
        picked.append(idx)
    return sorted(picked)


def allocate(counts: list[int], total_eval: int) -> list[int]:
    """홀드아웃 총 개수를 세션 크기에 비례해 나눈다 (최대잔여법).

    비례로 나누는 이유는 검증 손실이 세션들의 평균을 재게 하기 위해서다. 세션마다 같은
    개수를 빼면 작은 세션이 과대표되어, 실제로는 드문 조건이 점수를 좌우한다.
    """
    total = sum(counts)
    exact = [c * total_eval / total for c in counts]
    base = [int(math.floor(x)) for x in exact]
    remainder = total_eval - sum(base)
    order = sorted(range(len(counts)), key=lambda i: exact[i] - base[i], reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    for i, b in enumerate(base):
        if b == 0:
            raise SystemExit(
                f"세션 하나에 홀드아웃이 0회로 배정됐습니다(세션 크기 {counts[i]}회). "
                "eval_split을 올리거나 그 세션을 빼세요 — 검증이 비면 큐가 학습을 시작하지 않습니다."
            )
    return base


def read_info(root: Path, name: str) -> dict:
    info_path = root / name / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"데이터셋을 찾지 못했습니다: {info_path}")
    return json.loads(info_path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="만들 합본 데이터셋 이름")
    ap.add_argument(
        "--session",
        action="append",
        required=True,
        type=parse_session,
        metavar="이름=데이터셋[:검증인덱스]",
        help="합칠 세션. 여러 번 줄 수 있고, 준 순서가 학습 구간의 순서가 됩니다",
    )
    ap.add_argument("--eval-split", type=float, default=0.1, help="학습 때 쓸 eval_split (기본 0.1)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="데이터셋이 있는 폴더")
    ap.add_argument(
        "--keep-all-columns",
        action="store_true",
        help="센서 열을 전부 남깁니다. 기본은 정책이 실제로 먹는 열만 남기는 것입니다 — 아래 설명 참고",
    )
    ap.add_argument("--dry-run", action="store_true", help="구성만 계산해 보여 주고 만들지 않습니다")
    args = ap.parse_args()

    root: Path = args.root.expanduser()
    out_dir = root / args.out
    if out_dir.exists() and not args.dry_run:
        raise SystemExit(f"이미 있습니다: {out_dir}\n덮어쓰지 않습니다 — 다른 이름을 쓰거나 먼저 옮기세요.")

    sessions = args.session
    tags = [t for t, _, _ in sessions]
    if len(set(tags)) != len(tags):
        raise SystemExit("세션 이름이 겹칩니다")

    infos = [read_info(root, name) for _, name, _ in sessions]
    counts = [int(i["total_episodes"]) for i in infos]
    total = sum(counts)
    n_eval = math.ceil(total * args.eval_split)

    # 세션들이 같은 로봇·같은 카메라·같은 fps인지 본다. 다르면 합쳐도 학습이 죽거나,
    # 더 나쁘게는 조용히 이상해진다.
    ref = infos[0]
    for (tag, name, _), info in zip(sessions[1:], infos[1:], strict=False):
        for key in ("fps", "robot_type"):
            if info.get(key) != ref.get(key):
                raise SystemExit(f"{tag}({name})의 {key}가 첫 세션과 다릅니다: {info.get(key)} vs {ref.get(key)}")
        if set(info["features"]) != set(ref["features"]):
            only_a = sorted(set(ref["features"]) - set(info["features"]))
            only_b = sorted(set(info["features"]) - set(ref["features"]))
            raise SystemExit(f"{tag}({name})의 열 구성이 다릅니다.\n  첫 세션에만: {only_a}\n  이 세션에만: {only_b}")

    allocated = allocate(counts, n_eval)

    plan = []
    for (tag, name, explicit), n, k in zip(sessions, counts, allocated, strict=True):
        if explicit is not None:
            if len(explicit) != k:
                raise SystemExit(
                    f"{tag}: 직접 준 검증 인덱스가 {len(explicit)}개인데 이 구성에서 배정된 것은 {k}개입니다.\n"
                    f"  (전체 {total}회 x eval_split {args.eval_split} = {n_eval}회를 세션 크기에 비례해 나눈 값)"
                )
            if max(explicit) >= n:
                raise SystemExit(f"{tag}: 검증 인덱스 {max(explicit)}가 세션 크기 {n}회를 넘습니다")
            eval_idx = explicit
        else:
            eval_idx = spread(n, k)
        train_idx = [i for i in range(n) if i not in set(eval_idx)]
        plan.append({"tag": tag, "name": name, "n": n, "eval": eval_idx, "train": train_idx})

    train_total = sum(len(p["train"]) for p in plan)
    print(f"합본: {args.out} — 세션 {len(plan)}개 · 전체 {total}회 · 검증 {n_eval}회 (eval_split {args.eval_split})")
    for p in plan:
        print(f"  {p['tag']:10s} {p['name']:32s} {p['n']:3d}회 → 학습 {len(p['train']):3d} · 검증 {len(p['eval'])} {p['eval']}")
    print(f"  검증 자리: {train_total}~{total - 1} (맨 뒤 {n_eval}개)")

    if args.dry_run:
        print("\n--dry-run 이라 만들지 않았습니다.")
        return 0

    # lerobot은 무겁다. 계획을 다 세우고 검사를 통과한 뒤에 부른다.
    from lerobot.datasets.dataset_tools import merge_datasets, modify_features, split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    tmp = root / ".merge_tmp" / args.out
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    try:
        parts: dict[str, LeRobotDataset] = {}
        for p in plan:
            print(f"\n[{p['tag']}] 나누는 중 — 학습 {len(p['train'])} / 검증 {len(p['eval'])}")
            src = LeRobotDataset(repo_id=p["name"], root=root / p["name"])
            got = split_dataset(
                src,
                {f"{p['tag']}_train": p["train"], f"{p['tag']}_eval": p["eval"]},
                output_dir=tmp / p["tag"],
            )
            parts.update(got)

        order = [f"{p['tag']}_train" for p in plan] + [f"{p['tag']}_eval" for p in plan]
        print(f"\n합치는 중 — 순서: {' → '.join(order)}")
        merge_dir = out_dir if args.keep_all_columns else tmp / "merged"
        merged = merge_datasets([parts[k] for k in order], output_repo_id=args.out, output_dir=merge_dir)

        if not args.keep_all_columns:
            drop = sorted(
                k
                for k in merged.meta.features
                if k.startswith("observation.")
                and k != "observation.state"
                and not k.startswith("observation.images.")
            )
            print(f"\n정책이 안 먹는 열을 빼는 중 ({len(drop)}개): {drop}")
            modify_features(merged, remove_features=drop, output_dir=out_dir, repo_id=args.out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    merged_total = int(json.loads((out_dir / "meta" / "info.json").read_text())["total_episodes"])
    if merged_total != total:
        raise SystemExit(f"합본이 {merged_total}회입니다 — {total}회여야 합니다. 만들다 만 것을 지우세요: {out_dir}")

    record = {
        "order": order,
        "sessions": {
            p["tag"]: {
                "source": p["name"],
                "episodes": p["n"],
                "eval_indices": p["eval"],
                "train_count": len(p["train"]),
            }
            for p in plan
        },
        "total_episodes": total,
        "eval_count": n_eval,
        "eval_split": args.eval_split,
        "eval_range": [train_total, total - 1],
        "built_by": "sparkq/bin/soarm_merge_sessions.py",
        "note": (
            "lerobot은 과제마다 마지막 ceil(n x eval_split)개를 검증으로 뺀다. 그래서 합칠 때 "
            "세션별 홀드아웃을 맨 뒤에 모아 둔다. eval_range가 그 자리다. eval_indices는 "
            "합치기 전 각 세션 안에서의 번호다."
        ),
    }
    (out_dir / "SESSIONS.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    # 큐가 학습 전에 하는 검사를 여기서 한 번 더 한다. 여기서 통과하지 못하는 합본을
    # 만들어 두면 사람은 몇 시간 뒤 학습이 시작조차 안 한 것을 보게 된다.
    declared = sum(len(v["eval_indices"]) for v in record["sessions"].values())
    problems = []
    if declared != math.ceil(total * args.eval_split):
        problems.append(f"선언된 검증 {declared}회 ≠ lerobot이 뺄 {math.ceil(total * args.eval_split)}회")
    if record["eval_range"] != [total - n_eval, total - 1]:
        problems.append(f"검증 자리가 맨 뒤가 아닙니다: {record['eval_range']}")
    empty = [k for k, v in record["sessions"].items() if not v["eval_indices"]]
    if empty:
        problems.append("검증이 하나도 없는 세션: " + ", ".join(empty))
    if problems:
        raise SystemExit("만들어 놓고 보니 큐의 검사를 통과하지 못합니다:\n  - " + "\n  - ".join(problems))

    print(f"\n됐습니다: {out_dir}")
    print("[검사] " + " + ".join(f"{k} {len(v['eval_indices'])}회" for k, v in record["sessions"].items())
          + f" = {n_eval}회 (전체 {total}회 중 {record['eval_range'][0]}~{record['eval_range'][1]}번)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
