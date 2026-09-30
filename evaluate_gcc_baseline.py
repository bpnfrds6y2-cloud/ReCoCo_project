#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_gcc_baseline.py

GCCベースライン評価スクリプト。

denama/ReCoCo(論文公式リポジトリ)の apply_model/BandwidthEstimator_gcc.py
にある GCCEstimator (Google Congestion Control のPython完全再実装。学習
不要のヒューリスティック)を、SAC/TD3と同じ9 traceに対してそのまま走らせる。

出力する指標は2種類:
  1. mean_reward等 -- 既存のSAC_*/TD3_*_eval.csvと同じ列。論文のreward式
     (Ru, Rd, Rl, 式3-6)をそのまま実装しているので直接比較できる。
  2. qoe_receiving_rate / qoe_delay / qoe_loss / overall_qoe -- 論文Table I
     と同じQoEスコア(式7-10)。

学習不要・GPU不要(CPU評価のみ)。GCCは決定論的なので1回の実行で十分。

使い方(Dockerコンテナ内):
    docker compose exec pytorch python3 evaluate_gcc_baseline.py \
        --traces-dir ./traces --out ./gcc_eval
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from gym_folder.alphartc_gym import gym_file
from BandwidthEstimator_gcc import GCCEstimator

STEP_TIME_MS = 200  # 論文のΔt


def reward_ru(u):
    if u <= 0 or u > 1:
        return -1.0
    if u <= 0.65:
        return 1.538 * u - 1
    return -8.2 * (u - 1) ** 2 + 1


def reward_rd(d_ms):
    if d_ms <= 150:
        return -0.00667 * d_ms + 1
    if d_ms <= 200:
        return -0.02 * d_ms + 3
    return -1.0


def reward_rl(l):
    if l <= 0.02:
        return 1.0
    if l <= 0.1:
        return 156 * (l - 0.1) ** 2
    if l <= 0.2:
        return 100 * (l - 0.2) ** 2 - 1
    return -1.0


def calculate_reward(u, d_ms, l):
    ru, rd, rl = reward_ru(u), reward_rd(d_ms), reward_rl(l)
    r = (0.333 * ru + 0.333 * rd + 0.333 * rl) if l > 0 else (0.4 * ru + 0.4 * rd + 0.2 * rl)
    if l < 0.02 and d_ms < 30 and u > 0.9:
        r = 1.0
    return r


def build_bandwidth_lookup(trace_path):
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pattern = data["uplink"]["trace_pattern"]
    boundaries = []
    t = 0.0
    for seg in pattern:
        t += float(seg["duration"])
        boundaries.append((t, float(seg["capacity"])))

    def capacity_at(elapsed_ms):
        for end_ms, cap in boundaries:
            if elapsed_ms <= end_ms:
                return cap
        return boundaries[-1][1]

    return capacity_at


def evaluate_gcc_on_trace(trace_path, max_steps=20000):
    capacity_at = build_bandwidth_lookup(trace_path)

    gym_env = gym_file.Gym()
    gym_env.reset(trace_path=trace_path, report_interval_ms=STEP_TIME_MS, duration_time_ms=0)

    bwe = GCCEstimator()
    bwe.reset()
    bandwidth_prediction, _ = bwe.get_estimated_bandwidth()

    per_step_util, per_step_delay_ms, per_step_loss, per_step_reward = [], [], [], []
    elapsed_ms = 0.0

    for _ in range(max_steps):
        packet_list, done = gym_env.step(int(bandwidth_prediction))
        for pkt in packet_list:
            bwe.report_states(pkt)

        bandwidth_prediction, _ = bwe.get_estimated_bandwidth()

        receiving_rate_bps = bwe.packet_record.calculate_receiving_rate(interval=STEP_TIME_MS)
        delay_ms = bwe.packet_record.calculate_average_delay(interval=STEP_TIME_MS)
        loss_ratio = bwe.packet_record.calculate_loss_ratio(interval=STEP_TIME_MS)

        elapsed_ms += STEP_TIME_MS
        capacity_kbps = capacity_at(elapsed_ms)
        u = (receiving_rate_bps / 1000.0) / capacity_kbps if capacity_kbps > 0 else 0.0

        per_step_util.append(u)
        per_step_delay_ms.append(delay_ms)
        per_step_loss.append(loss_ratio)
        per_step_reward.append(calculate_reward(u, delay_ms, loss_ratio))

        if done:
            break

    return {
        "mean_reward": float(np.mean(per_step_reward)) if per_step_reward else 0.0,
        "mean_bandwidth_util": float(np.mean(per_step_util)) if per_step_util else 0.0,
        "mean_delay_ms": float(np.mean(per_step_delay_ms)) if per_step_delay_ms else 0.0,
        "mean_loss_ratio": float(np.mean(per_step_loss)) if per_step_loss else 0.0,
        "n_steps": len(per_step_reward),
        "_util": per_step_util,
        "_delay_ms": per_step_delay_ms,
        "_loss": per_step_loss,
    }


def qoe_from_raw(util, delay_ms, loss):
    u = np.clip(np.asarray(util, dtype=float), None, 1.0)
    qoe_rr = 100.0 * float(np.median(u))

    d = np.asarray(delay_ms, dtype=float)
    d_max, d_min, d95 = float(d.max()), float(d.min()), float(np.percentile(d, 95))
    qoe_delay = 100.0 if d_max == d_min else 100.0 * (d_max - d95) / (d_max - d_min)

    l = np.asarray(loss, dtype=float)
    qoe_loss = 100.0 * (1.0 - float(l.mean()))

    overall = 0.33 * qoe_rr + 0.33 * qoe_delay + 0.33 * qoe_loss
    return qoe_rr, qoe_delay, qoe_loss, overall


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces-dir", default="./traces")
    ap.add_argument("--out", default="./gcc_eval")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for trace_path in sorted(glob.glob(os.path.join(args.traces_dir, "*.json"))):
        trace_name = os.path.basename(trace_path)[:-5]
        print(f"[gcc] evaluating on {trace_name} ...")
        stats = evaluate_gcc_on_trace(trace_path)
        qoe_rr, qoe_delay, qoe_loss, overall_qoe = qoe_from_raw(
            stats["_util"], stats["_delay_ms"], stats["_loss"])

        row = {
            "trace": trace_name,
            "mean_reward": stats["mean_reward"],
            "mean_bandwidth_util": stats["mean_bandwidth_util"],
            "mean_delay_ms": stats["mean_delay_ms"],
            "mean_loss_ratio": stats["mean_loss_ratio"],
            "n_steps": stats["n_steps"],
            "qoe_receiving_rate": qoe_rr,
            "qoe_delay": qoe_delay,
            "qoe_loss": qoe_loss,
            "overall_qoe": overall_qoe,
        }
        rows.append(row)
        print(f"       reward={row['mean_reward']:.3f}  overall_QoE={row['overall_qoe']:.2f}")

    out_csv = os.path.join(args.out, "GCC_eval.csv")
    fieldnames = ["trace", "mean_reward", "mean_bandwidth_util", "mean_delay_ms",
                  "mean_loss_ratio", "n_steps", "qoe_receiving_rate", "qoe_delay",
                  "qoe_loss", "overall_qoe"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
