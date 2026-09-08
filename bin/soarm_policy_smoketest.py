#!/usr/bin/env python3
"""정책 하나를 큐에 걸기 전에 CPU에서 한 번 세워 보고 배치를 통과시킨다.

왜 있는가
---------
2026-09-08에 GR00T를 걸면서 큐 자리를 두 번 버렸다. 두 번 다 몇십 초 만에 죽었고, 두 번 다
GPU가 전혀 필요 없는 이유였다.

1. `diffusers`가 없어서 `make_policy`에서 ImportError — 6초.
2. 백본 토크나이저를 게이트된 저장소에서 받으려다 403 — 42초, **첫 학습 스텝의 전처리에서**.

두 번째가 중요하다. 그때 나는 모델을 CPU에서 세워 보는 확인은 했지만 **배치를 전처리기에
통과시키지는 않았다.** 모델이 서는 것과 배치가 지나가는 것은 다른 사건이고, 토크나이저·
프로세서·정규화 통계는 뒤쪽에서만 만져진다. 그래서 이 스크립트는 둘 다 한다.

`CUDA_VISIBLE_DEVICES`를 비워서 GPU를 아예 볼 수 없게 만든 뒤에 돈다 — 큐 밖에서 돌려도
되는 이유가 그것이다. 확인하고 나서 sparkq에 건다.

쓰는 법
-------
    python3 soarm_policy_smoketest.py --dataset soarm101_cube134_dnv_strat --policy groot

`--policy`는 `kinds/lerobot-train.json`의 정책 이름이다. 그 프리셋의 `flag`를 읽어 같은
설정으로 세우므로, 큐가 걸 것과 같은 물건을 본다.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

# 무엇보다 먼저 — torch를 들이기 전에 GPU를 가린다.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

KINDS = Path.home() / "sparkq" / "kinds" / "lerobot-train.json"
DATASET_ROOT = Path.home() / "data" / "soarm"


def preset_flags(policy: str) -> dict[str, str]:
    """종류 파일에서 그 정책의 flag를 읽어 --policy.* 인자만 뜯어낸다."""
    doc = json.loads(KINDS.read_text())
    field = next(f for f in doc["fields"] if f["name"] == "policy")
    if policy not in field["presets"]:
        raise SystemExit(f"모르는 정책입니다: {policy}. 있는 것: {field['values']}")
    flag = field["presets"][policy]["flag"]
    out: dict[str, str] = {}
    for token in shlex.split(flag):
        if token.startswith("--policy.") and "=" in token:
            key, value = token.split("=", 1)
            out[key.removeprefix("--policy.")] = value
    return out


def coerce(value: str):
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--forward", action="store_true", help="전처리에 더해 손실까지 한 번 구합니다 (느립니다)")
    args = ap.parse_args()

    import torch

    if torch.cuda.is_available():
        raise SystemExit("GPU가 보입니다 — CUDA_VISIBLE_DEVICES가 비워지지 않았습니다. 멈춥니다.")

    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import dataset_to_policy_features, get_policy_class, make_pre_post_processors

    root = args.root / args.dataset
    if not root.exists():
        raise SystemExit(f"데이터셋이 없습니다: {root}")

    flags = preset_flags(args.policy)
    ptype = flags.pop("type", None)
    if ptype is None:
        raise SystemExit(
            f"'{args.policy}' 프리셋은 --policy.path로 사전학습을 불러옵니다. 그 경로는 config가 입력을 "
            "덮어쓰므로 이 점검의 대상이 아닙니다 — 이 스크립트는 --policy.type 경로(groot, act)를 봅니다."
        )

    print(f"데이터셋 {args.dataset} · 정책 {args.policy}(type={ptype})")
    dataset = LeRobotDataset(repo_id=args.dataset, root=root)
    feats = dataset_to_policy_features(dataset.meta.features)

    from lerobot.configs.policies import PreTrainedConfig

    cfg_cls = PreTrainedConfig.get_choice_class(ptype)
    cfg = cfg_cls(**{k: coerce(v) for k, v in flags.items()})
    cfg.device = "cpu"
    cfg.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    cfg.input_features = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}
    cfg.validate_features()
    print("  특징 계약 통과 —", sorted(cfg.input_features), "→", sorted(cfg.output_features))

    print("  모델 세우는 중 (CPU)…")
    policy = get_policy_class(ptype)(cfg, dataset_stats=dataset.meta.stats)
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  세워짐 — 전체 {total / 1e9:.2f}B · 학습 {trainable / 1e9:.2f}B ({100 * trainable / total:.0f}%)")

    # 여기부터가 2026-09-08에 놓쳤던 자리다. 토크나이저·프로세서·통계는 배치가 지날 때 만져진다.
    print("  전처리기에 배치 한 개 통과시키는 중…")
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)))
    batch = preprocessor(batch)
    print("  전처리 통과 — 배치 열:", sorted(k for k in batch if not k.startswith("_"))[:8], "…")

    if args.forward:
        print("  손실 한 번 구하는 중 (CPU라 느립니다)…")
        policy.train()
        loss, _ = policy.forward(batch)
        print(f"  손실 {float(loss):.4f}")

    print("\n걸어도 됩니다:")
    print(f"  python3 ~/sparkq/sparkq.py add lerobot-train dataset={args.dataset} policy={args.policy} validation=on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
