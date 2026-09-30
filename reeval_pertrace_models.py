#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reeval_pertrace_models.py

evaluate_on_traceにQoEパッチを当てた後、既存の学習済み18個の
per-traceモデル(sb3_models/{SAC,TD3}_{TRACE}.zip)を再学習なしで
再評価し、*_eval.csvをQoE列込みで上書きする。CPUで実行可能。

使い方(Dockerコンテナ内):
    docker compose exec pytorch python3 reeval_pertrace_models.py
"""
import glob
import os

from stable_baselines3 import SAC, TD3

from sb3_recoco_reproduction import evaluate_on_trace, save_eval_csv


def main():
    for path in sorted(glob.glob("./sb3_models/*.zip")):
        name = os.path.basename(path)[:-4]
        if name.endswith("_global"):
            continue  # globalモデルは別途、学習後の自動評価で再生成する
        agent = "SAC" if name.startswith("SAC") else "TD3"
        trace_name = name[len(agent) + 1:]
        trace_path = f"./traces/{trace_name}.json"
        if not os.path.exists(trace_path):
            print(f"[skip] trace not found for {name}: {trace_path}")
            continue

        cls = SAC if agent == "SAC" else TD3
        model = cls.load(path, device="cpu")
        print(f"[reeval] {name} on {trace_path} ...")
        result = evaluate_on_trace(model, trace_path, n_episodes=3)
        save_eval_csv([result], path)
        print(f"  -> overall_QoE={result['overall_qoe']:.2f}  mean_reward={result['mean_reward']:.4f}")


if __name__ == "__main__":
    main()
