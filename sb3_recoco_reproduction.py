#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sb3_recoco_reproduction.py

ReCoCo論文(denama/ReCoCo, IEEE HPSR 2023)をできる限り忠実に再現するための
stable-baselines3ベースの学習スクリプト。GPU機(このMacではない別マシン)で
実行することを想定している。

== このスクリプトが再現しているもの ==

1. アルゴリズム/ハイパーパラメータ: 本家の conf_dict_params.py の
   hyperparams_SAC / hyperparams_TD3 (tuned=True 相当)をそのまま採用:

     SAC: lr=3e-4, buffer_size=50000, batch_size=512, tau=0.01, gamma=0.9999,
          train_freq=32, gradient_steps=32, ent_coef=0.1(固定、自動調整なし),
          use_sde=True, net_arch=[64,64], log_std_init=-3.67, learning_starts=0
     TD3: gamma=0.98, learning_starts=10000, action_noise=Normal(sigma=0.1),
          (他はSB3のTD3デフォルトのまま。本家の train_new_data.py で実際に
          "アクティブ"だった alg="TD3", tuned=False の場合はSB3デフォルト
          そのまま+action_noiseのみ、というのが本家の実運用設定でもあった。
          --tuned false でこちらに切り替え可能)

2. 報酬関数: 本家 rtc_env.py の calculate_reward() を一字一句再現した
   OriginalRewardGymEnv サブクラスを使う(このプロジェクトの rtc_env.py は
   Ru項の式や崖ペナルティを独自に改良済みのため、論文再現の目的では
   そちらを使わず、このファイル内で完全に元の式に戻している)。

3. trace の与え方: 本家と同じく random_trace=True で、エピソード終了
   (=trace終了)ごとに trace_set からランダムに次のtraceを選ぶ。
   1つのモデル/リプレイバッファ/オプティマイザを最初から最後まで
   連続して学習させる(pretrain_global_agent()のような固定順の
   trace-by-traceブロック学習はしない)。

== 生成するモデルの構成(全20モデル) ==

  各アルゴリズム(SAC, TD3)ごとに:
    - trace個別モデル 9個(--mode pertrace、traceを1本だけ渡す、
      random_trace=False。「事前にtraceごとに個別学習したモデルを
      取捨選択して使う」という、この研究が非現実的だとしている
      比較対象/ベースラインを作るためのもの)
    - 汎用(グローバル)モデル 1個(--mode global、9trace全部を
      trace_setに渡し random_trace=True。これがこの研究でいう
      「グローバルモデル」= FLで発展させていく出発点)
  (9+1) x 2 = 20モデル

== 使い方 ==

  # trace個別モデル(9trace分、SAC): 1本ずつ実行
  python3 sb3_recoco_reproduction.py --agent-type SAC --mode pertrace \\
      --trace ./traces/4G_3mbps.json --total-timesteps 200000 \\
      --output ./sb3_models/SAC_4G_3mbps.zip

  # 汎用(グローバル)モデル(SAC、9trace全部からランダムサンプリング)
  python3 sb3_recoco_reproduction.py --agent-type SAC --mode global \\
      --total-timesteps 750000 \\
      --output ./sb3_models/SAC_global.zip

  # TD3も同様に --agent-type TD3 で実行(下記の run_all_20.sh で一括実行可能)

GPU使用: --device cuda を指定(未指定時は自動判定 = "auto"、CUDAが
使えればGPUを使う)。ネットワークが小さいため速度向上は限定的だが、
指定しても害はない。
"""
import argparse
import os

import numpy as np

from rtc_env import GymEnv


# ============================================================
# 原論文(denama/ReCoCo)の calculate_reward() を一字一句再現したサブクラス。
# このプロジェクトの rtc_env.py 本体の calculate_reward() は独自改良済み
# (Ru項をべき乗ランプに変更、delay>1000ms/loss>0.2の崖ペナルティを撤廃)
# のため、論文再現の目的ではこちらを使う。rtc_env.py 自体は一切変更しない。
# ============================================================
class OriginalRewardGymEnv(GymEnv):
    def calculate_reward(self):
        sending_rate = self.sending_rate / 1000  # bps -> kbps
        bandwidth = self.current_bandwidth        # 既にkbps
        receiving_rate = self.receiving_rate / 1000
        delay = self.delay
        loss_ratio = self.loss_ratio

        if bandwidth <= 0.00001:
            bandwidth_util = 0
        else:
            bandwidth_util = receiving_rate / bandwidth

        # 物理的にありえない値 -> reward=-1
        if (delay < 0) or (loss_ratio > 1) or (loss_ratio < 0) \
                or (bandwidth_util < 0) or (bandwidth_util > 1):
            return -1

        # 原論文の"崖"ペナルティ(このプロジェクトのrtc_env.pyでは撤廃済みだが
        # 論文再現のためここでは残す)
        if (receiving_rate > bandwidth) or (receiving_rate > sending_rate) \
                or (delay > 1000) or (loss_ratio > 0.2):
            return -1

        if (loss_ratio < 0.02) and (delay < 30) and (bandwidth_util > 0.9):
            return 1

        # Ru: 利用率項(reward_profile=0、原論文のオリジナルの線形+二次式)
        threshold = 0.65
        if 0 <= bandwidth_util <= threshold:
            Ru = (1.538 * bandwidth_util) - 1
        else:
            Ru = -8.2 * ((bandwidth_util - 1) ** 2) + 1

        # Rd: 遅延項
        if 0 <= delay <= 150:
            Rd = -0.00667 * delay + 1
        elif 150 < delay <= 200:
            Rd = -0.02 * delay + 3
        else:
            Rd = -1
        Rd = round(Rd, 4)

        # Rl: ロス項
        if 0 <= loss_ratio <= 0.02:
            Rl = 1
        elif 0.02 < loss_ratio <= 0.1:
            Rl = 156 * ((loss_ratio - 0.1) ** 2)
        elif 0.1 < loss_ratio <= 0.2:
            Rl = 100 * ((loss_ratio - 0.2) ** 2) - 1
        else:
            Rl = -1
        Rl = round(Rl, 4)

        if loss_ratio > 0:
            reward = 0.333 * Ru + 0.333 * Rd + 0.333 * Rl
        else:
            reward = 0.4 * Ru + 0.4 * Rd + 0.2 * Rl
        return reward


DEFAULT_TRACE_SET = [
    "./traces/4G_3mbps.json",
    "./traces/4G_500kbps.json",
    "./traces/4G_700kbps.json",
    "./traces/5G_12mbps.json",
    "./traces/5G_13mbps.json",
    "./traces/WIRED_200kbps.json",
    "./traces/WIRED_35mbps.json",
    "./traces/WIRED_900kbps.json",
    "./traces/trace_300k.json",
]


def build_env(mode, trace, trace_set):
    if mode == "pertrace":
        if not trace:
            raise ValueError("--mode pertrace のときは --trace で対象traceを1つ指定してください")
        return OriginalRewardGymEnv(
            step_time=200,
            input_trace=trace,
            random_trace=False,
            normalize_states=True,
            delay_states=True,
        )
    elif mode == "global":
        ts = trace_set or DEFAULT_TRACE_SET
        missing = [t for t in ts if not os.path.exists(t)]
        if missing:
            raise FileNotFoundError(f"trace_setに存在しないファイルがあります: {missing}")
        return OriginalRewardGymEnv(
            step_time=200,
            trace_set=ts,
            random_trace=True,
            normalize_states=True,
            delay_states=True,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")


def build_model(agent_type, env, tuned, seed, device, tensorboard_log):
    if agent_type == "SAC":
        from stable_baselines3 import SAC
        if tuned:
            return SAC(
                policy="MlpPolicy", env=env, seed=seed, device=device,
                learning_rate=3e-4, buffer_size=50_000, learning_starts=0,
                batch_size=512, tau=0.01, gamma=0.9999,
                train_freq=32, gradient_steps=32, ent_coef=0.1,
                use_sde=True, policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]),
                tensorboard_log=tensorboard_log, verbose=1,
            )
        else:
            return SAC(policy="MlpPolicy", env=env, seed=seed, device=device,
                        tensorboard_log=tensorboard_log, verbose=1)
    elif agent_type == "TD3":
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        if tuned:
            return TD3(
                policy="MlpPolicy", env=env, seed=seed, device=device,
                gamma=0.98, learning_starts=10_000, action_noise=action_noise,
                tensorboard_log=tensorboard_log, verbose=1,
            )
        else:
            # 本家 train_new_data.py で実際にアクティブだった設定
            # (alg="TD3", tuned=False): SB3のデフォルト + action_noiseのみ
            return TD3(policy="MlpPolicy", env=env, action_noise=action_noise,
                       seed=seed, device=device, tensorboard_log=tensorboard_log, verbose=1)
    else:
        raise ValueError(f"unknown agent_type: {agent_type}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-type", choices=["SAC", "TD3"], required=True)
    p.add_argument("--mode", choices=["pertrace", "global"], required=True)
    p.add_argument("--trace", type=str, default=None,
                    help="--mode pertrace のとき: 対象trace(例: ./traces/4G_3mbps.json)")
    p.add_argument("--trace-set", nargs="+", default=None,
                    help="--mode global のとき: 省略時は9trace全部(DEFAULT_TRACE_SET)")
    p.add_argument("--total-timesteps", type=int, required=True)
    p.add_argument("--tuned", dest="tuned", action="store_true", default=True,
                    help="本家のtuned=Trueハイパーパラメータを使う(デフォルト)")
    p.add_argument("--no-tuned", dest="tuned", action="store_false",
                    help="本家のtuned=False(SB3デフォルト+action noiseのみ)を使う")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                    help="'cuda' / 'cpu' / 'auto'(自動判定、デフォルト)")
    p.add_argument("--output", type=str, required=True,
                    help="保存先(.zip、SB3標準形式)")
    p.add_argument("--tensorboard-log", type=str, default=None,
                    help="指定すると学習曲線をtensorboardログとして保存")
    args = p.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    env = build_env(args.mode, args.trace, args.trace_set)
    model = build_model(args.agent_type, env, args.tuned, args.seed,
                         args.device, args.tensorboard_log)

    label = args.trace if args.mode == "pertrace" else "GLOBAL(9trace random)"
    print(f"[sb3_recoco_reproduction] agent={args.agent_type} mode={args.mode} "
          f"target={label} tuned={args.tuned} total_timesteps={args.total_timesteps} "
          f"device={args.device}")

    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    model.save(args.output)
    print(f"[sb3_recoco_reproduction] 保存しました: {args.output}")


if __name__ == "__main__":
    main()
