#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_test.py

GPU機で本番の20モデル学習を始める前に、環境が正しく動くかを安く確認する
ためのスクリプト。以下を数秒〜数十秒で検証する:
  1. gym / torch / stable_baselines3 が正しくimportできるか
  2. torch.cuda.is_available() が True か(GPUを使う予定なら)
  3. NS3+WebRTCの事前コンパイル済みバイナリ(gym_folder/target)が
     このコンテナ内でも起動できるか(env.reset()/env.step()が動くか)
  4. sb3_recoco_reproduction.py の OriginalRewardGymEnv でSACモデルを
     ごく短く(1000ステップ)学習できるか

使い方:
    python3 smoke_test.py
"""
import sys

print("=== 1. import check ===")
import gym
print(f"gym version: {gym.__version__}")
import torch
print(f"torch version: {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
import stable_baselines3
print(f"stable_baselines3 version: {stable_baselines3.__version__}")

print("\n=== 2. rtc_env / NS3 gym binary check ===")
from rtc_env import GymEnv
env = GymEnv(step_time=200, input_trace="./traces/trace_300k.json",
             random_trace=False, normalize_states=True, delay_states=True)
obs = env.reset()
print(f"reset OK, obs shape: {obs.shape}")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"  step {i}: reward={reward:.4f}, done={done}")
    if done:
        obs = env.reset()
env.close()
print("NS3 gym binary: OK (5ステップ実行できました)")

print("\n=== 3. sb3_recoco_reproduction.py の短時間学習テスト ===")
from sb3_recoco_reproduction import build_env, build_model
env2 = build_env("pertrace", "./traces/trace_300k.json", None)
model = build_model("SAC", env2, tuned=True, seed=0, device="auto", tensorboard_log=None)
model.learn(total_timesteps=1000, progress_bar=False)
print("SACモデルの短時間学習: OK")

print("\n=== 全チェック完了。本番の run_all_20.sh を実行して問題ありません ===")
