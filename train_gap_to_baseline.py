#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gap_to_baseline.py

論文(denama/ReCoCo)がgap-to-baseline方式と呼ぶカリキュラム学習の再現。
1つのモデル/リプレイバッファ/オプティマイザを、simple(GCCに大勝ちできる
=簡単)なtraceから難しいtraceへ、順番にブロックごとに連続学習させる。
論文でこの実験に使われたのはTD3・delayed states・untuned(SB3デフォルト+
action_noiseのみ)の1構成のみ(Section VI-C, Fig.8)。

学習順序は compute_curriculum_order.py が生成した curriculum_order.json
(このプロジェクト自身のGCC比較データから計算したもの)を使う。

使い方(Dockerコンテナ内、GPU使用):
    docker compose exec pytorch python3 train_gap_to_baseline.py \
        --trace-order-file ./curriculum_order.json \
        --steps-per-trace 300000 \
        --agent-type TD3 --no-tuned \
        --output ./sb3_models/TD3_global_gap.zip

  --steps-per-trace 300000 x 9trace = 2.7M steps(論文と同じ予算)。
  V100で1M stepsのTD3学習が4時間程度(論文本文の記述)なので、目安として
  2.7M stepsで11時間前後かかる可能性がある。GPU時間を抑えたい場合は
  この値を小さくする(例: 83000 で 750k steps相当、これまでのglobalモデル
  と同じ予算)。
"""
import argparse
import json
import os

import numpy as np

from sb3_recoco_reproduction import (
    build_env, build_model, run_evaluation, save_eval_csv, DEFAULT_TRACE_SET,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace-order-file", type=str, default="./curriculum_order.json",
                    help="compute_curriculum_order.py が出力したJSON"
                         "(trace jsonパスのリスト、簡単->難しい順)")
    p.add_argument("--steps-per-trace", type=int, default=300_000,
                    help="1traceブロックあたりの学習ステップ数(論文は300000)")
    p.add_argument("--agent-type", choices=["SAC", "TD3"], default="TD3",
                    help="論文のgap-to-baseline実験はTD3のみ(デフォルト)")
    p.add_argument("--tuned", dest="tuned", action="store_true", default=False)
    p.add_argument("--no-tuned", dest="tuned", action="store_false",
                    help="論文のこの実験はnot tuned(デフォルト)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--tensorboard-log", type=str, default="./tb_logs")
    p.add_argument("--eval-episodes", type=int, default=3)
    args = p.parse_args()

    with open(args.trace_order_file) as f:
        trace_order = json.load(f)
    missing = [t for t in trace_order if not os.path.exists(t)]
    if missing:
        raise FileNotFoundError(f"trace_orderに存在しないファイルがあります: {missing}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tb_log = args.tensorboard_log if args.tensorboard_log else None

    print(f"[gap-to-baseline] agent={args.agent_type} tuned={args.tuned} "
          f"steps_per_trace={args.steps_per_trace} "
          f"total={args.steps_per_trace * len(trace_order)} device={args.device}")
    print(f"[gap-to-baseline] 学習順序(簡単->難しい): {trace_order}")

    model = None
    for i, trace_path in enumerate(trace_order):
        env = build_env("pertrace", trace_path, None)
        reset_num_timesteps = (i == 0)

        if model is None:
            model = build_model(args.agent_type, env, args.tuned, args.seed,
                                 args.device, tb_log)
        else:
            model.set_env(env)

        run_name = f"{args.agent_type}_gap_block{i}_{os.path.basename(trace_path)}"
        print(f"\n[gap-to-baseline] === ブロック{i+1}/{len(trace_order)}: "
              f"{trace_path} ({args.steps_per_trace} steps) ===")
        if tb_log:
            print(f"[gap-to-baseline] tensorboard: {tb_log}/{run_name}_1")

        model.learn(total_timesteps=args.steps_per_trace, progress_bar=False,
                    reset_num_timesteps=reset_num_timesteps, tb_log_name=run_name)
        print(f"[gap-to-baseline] ブロック{i+1}完了。累計num_timesteps="
              f"{model.num_timesteps}")

    model.save(args.output)
    print(f"\n[gap-to-baseline] 保存しました: {args.output}")

    if args.eval_episodes > 0:
        print(f"[gap-to-baseline] 9trace全部で評価します"
              f"(trace毎に{args.eval_episodes}エピソード)...")
        results = run_evaluation(model, "global", None, DEFAULT_TRACE_SET,
                                  n_episodes=args.eval_episodes)
        save_eval_csv(results, args.output)
        overall_mean_reward = float(np.mean(
            [r["mean_reward"] for r in results if r["mean_reward"] is not None]))
        overall_mean_qoe = float(np.mean(
            [r["overall_qoe"] for r in results if r["overall_qoe"] is not None]))
        print(f"[gap-to-baseline] 全trace平均reward: {overall_mean_reward:.4f}  "
              f"全trace平均Overall QoE: {overall_mean_qoe:.2f}")


if __name__ == "__main__":
    main()
