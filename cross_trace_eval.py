#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cross_trace_eval.py

論文Fig.7相当のクロストレース評価。学習済みのper-traceモデル
(sb3_models/{AGENT}_{TRACE}.zip, globalモデルは除く)を、学習に使った
trace以外も含めた全9traceで評価する。再学習なし・CPUで実行可能。

使い方(Dockerコンテナ内):
    docker compose exec pytorch python3 cross_trace_eval.py \
        --models-dir ./sb3_models --out ./cross_trace_results.csv
"""
import argparse
import csv
import glob
import os

from stable_baselines3 import SAC, TD3

from sb3_recoco_reproduction import evaluate_on_trace, DEFAULT_TRACE_SET


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default="./sb3_models")
    ap.add_argument("--out", default="./cross_trace_results.csv")
    ap.add_argument("--n-episodes", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.models_dir, "*.zip"))):
        name = os.path.basename(path)[:-4]
        if name.endswith("_global"):
            continue  # globalモデルは既存の_eval.csvで9trace評価済みなのでスキップ
        agent = "SAC" if name.startswith("SAC") else "TD3"
        train_trace = name[len(agent) + 1:]
        cls = SAC if agent == "SAC" else TD3
        model = cls.load(path, device="cpu")

        for test_trace_path in DEFAULT_TRACE_SET:
            print(f"[cross] {name} -> {test_trace_path}")
            r = evaluate_on_trace(model, test_trace_path, n_episodes=args.n_episodes)
            r["agent"] = agent
            r["train_trace"] = train_trace
            rows.append(r)

    fieldnames = ["agent", "train_trace", "trace", "mean_reward", "mean_bandwidth_util",
                  "mean_delay_ms", "mean_loss_ratio", "n_steps",
                  "qoe_receiving_rate", "qoe_delay", "qoe_loss", "overall_qoe"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {args.out} ({len(rows)} rows = per-traceモデル数 x 9traces)")


if __name__ == "__main__":
    main()
