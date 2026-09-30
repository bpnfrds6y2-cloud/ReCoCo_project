#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_curriculum_order.py

論文のgap-to-baseline方式(「専用モデルがGCCに対してQoEでどれだけ勝てるか」
の差が大きいtrace=簡単、小さい/負けているtrace=難しい、として並べる)の
学習順序を、このプロジェクト自身のデータから計算する。

前提(どちらも再学習不要、先に実行しておくこと):
  1. reeval_pertrace_models.py を実行済み(sb3_models/TD3_*_eval.csv に
     overall_qoe列がある状態)
  2. evaluate_gcc_baseline.py を実行済み(./gcc_eval/GCC_eval.csv がある)

使い方:
    python3 compute_curriculum_order.py \
        --pertrace-dir ./sb3_models --gcc-csv ./gcc_eval/GCC_eval.csv \
        --agent TD3 --out ./curriculum_order.json
"""
import argparse
import csv
import json
import os


def read_gcc_qoe(gcc_csv):
    qoe = {}
    with open(gcc_csv, newline="") as f:
        for row in csv.DictReader(f):
            qoe[row["trace"]] = float(row["overall_qoe"])
    return qoe


def read_pertrace_qoe(pertrace_dir, agent, trace_names):
    qoe = {}
    for t in trace_names:
        path = os.path.join(pertrace_dir, f"{agent}_{t}_eval.csv")
        with open(path, newline="") as f:
            row = next(csv.DictReader(f))
            if row.get("overall_qoe") in (None, "", "None"):
                raise ValueError(
                    f"{path} にoverall_qoe列がありません。"
                    f"先に reeval_pertrace_models.py を実行してください。")
            qoe[t] = float(row["overall_qoe"])
    return qoe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pertrace-dir", default="./sb3_models")
    ap.add_argument("--gcc-csv", default="./gcc_eval/GCC_eval.csv")
    ap.add_argument("--agent", default="TD3",
                     help="gap計算に使うper-traceモデルのアルゴリズム"
                          "(論文のgap-to-baseline実験はTD3のみで行われた)")
    ap.add_argument("--out", default="./curriculum_order.json")
    args = ap.parse_args()

    trace_names = ["4G_3mbps", "4G_500kbps", "4G_700kbps", "5G_12mbps", "5G_13mbps",
                   "WIRED_200kbps", "WIRED_35mbps", "WIRED_900kbps", "trace_300k"]

    gcc_qoe = read_gcc_qoe(args.gcc_csv)
    pertrace_qoe = read_pertrace_qoe(args.pertrace_dir, args.agent, trace_names)

    gaps = []
    for t in trace_names:
        gap = pertrace_qoe[t] - gcc_qoe[t]
        gaps.append((t, gap, pertrace_qoe[t], gcc_qoe[t]))

    # 差が大きい(=GCCに大勝ち=簡単)順にソート -> 難しいtraceは最後
    gaps.sort(key=lambda x: x[1], reverse=True)

    print(f"{'trace':<16} {'gap(specialized-GCC)':>22} {'specialized_QoE':>18} {'GCC_QoE':>10}")
    for t, gap, sq, gq in gaps:
        print(f"{t:<16} {gap:>22.2f} {sq:>18.2f} {gq:>10.2f}")

    order = [f"./traces/{t}.json" for t, _, _, _ in gaps]
    with open(args.out, "w") as f:
        json.dump(order, f, indent=2, ensure_ascii=False)
    print(f"\n[ok] 学習順序(簡単->難しい)を {args.out} に保存しました")
    print(json.dumps(order, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
