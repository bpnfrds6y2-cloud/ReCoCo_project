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

== 結果の確認方法(2通り) ==

1. 学習中の途中経過(tensorboard): デフォルトで ./tb_logs/ にログを書く。
   別ターミナルで
       tensorboard --logdir ./tb_logs --host 0.0.0.0
   を実行し、http://<GPU機のIP>:6006 にブラウザでアクセスすると
   reward等の学習曲線をリアルタイムで見られる(--tensorboard-log '' で無効化可)。

2. 学習後の評価結果(数値): 学習完了後、自動的に対象trace(pertraceなら
   そのtrace、globalなら9trace全部)で決定論的に評価を行い、
   <output>_eval.csv に mean_reward / mean_bandwidth_util / mean_delay_ms /
   mean_loss_ratio を保存する(--eval-episodes 0 でスキップ可)。
"""
import argparse
import csv
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


def evaluate_on_trace(model, trace_path, n_episodes=3):
    """学習済みモデルを指定traceで決定論的に走らせ、reward・帯域利用率・遅延・
    ロス率の平均値を計算する(_plot_pretrain_timeseries等、他のスクリプトが
    使っている window_mean_* と同じ指標)。「精度の結果」を数値で見るための
    評価関数。
    """
    env = OriginalRewardGymEnv(
        step_time=200, input_trace=trace_path, random_trace=False,
        normalize_states=True, delay_states=True,
    )
    rewards, utils, delays, losses = [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rewards.append(float(reward))
            bw = getattr(env, "current_bandwidth", 0)
            rr = getattr(env, "receiving_rate", 0) / 1000
            if bw > 0.00001:
                utils.append(min(rr / bw, 1.5))
            delays.append(float(getattr(env, "delay", 0.0)))
            losses.append(float(getattr(env, "loss_ratio", 0.0)))
    env.close()
    return dict(
        trace=trace_path,
        mean_reward=float(np.mean(rewards)) if rewards else None,
        mean_bandwidth_util=float(np.mean(utils)) if utils else None,
        mean_delay_ms=float(np.mean(delays)) if delays else None,
        mean_loss_ratio=float(np.mean(losses)) if losses else None,
        n_steps=len(rewards),
    )


def run_evaluation(model, mode, trace, trace_set, n_episodes=3):
    """--mode pertrace ならそのtrace1本、--mode global なら9trace全部について
    それぞれ evaluate_on_trace() を実行し、結果のリストを返す(globalモデルが
    trace別にどれくらい汎化できているかが分かる)。
    """
    if mode == "pertrace":
        targets = [trace]
    else:
        targets = trace_set or DEFAULT_TRACE_SET
    results = []
    for t in targets:
        print(f"[sb3_recoco_reproduction] 評価中: {t} ({n_episodes}エピソード)...")
        r = evaluate_on_trace(model, t, n_episodes=n_episodes)
        print(f"  -> mean_reward={r['mean_reward']:.4f}  "
              f"util={r['mean_bandwidth_util']:.3f}  "
              f"delay={r['mean_delay_ms']:.1f}ms  "
              f"loss={r['mean_loss_ratio']:.4f}")
        results.append(r)
    return results


def save_eval_csv(results, output_path):
    base, _ = os.path.splitext(output_path)
    csv_path = f"{base}_eval.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[sb3_recoco_reproduction] 評価結果を保存しました: {csv_path}")
    return csv_path


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
    p.add_argument("--tensorboard-log", type=str, default="./tb_logs",
                    help="学習曲線のtensorboardログ保存先ディレクトリ"
                         "(デフォルト./tb_logs。'tensorboard --logdir tb_logs'"
                         "で途中経過をブラウザから見られる。無効化するには"
                         "空文字 '' を指定)")
    p.add_argument("--eval-episodes", type=int, default=3,
                    help="学習後の評価に使うエピソード数(0で評価をスキップ)")
    args = p.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tb_log = args.tensorboard_log if args.tensorboard_log else None

    env = build_env(args.mode, args.trace, args.trace_set)
    model = build_model(args.agent_type, env, args.tuned, args.seed,
                         args.device, tb_log)

    label = args.trace if args.mode == "pertrace" else "GLOBAL(9trace random)"
    run_name = f"{args.agent_type}_{args.mode}_{os.path.splitext(os.path.basename(args.output))[0]}"
    print(f"[sb3_recoco_reproduction] agent={args.agent_type} mode={args.mode} "
          f"target={label} tuned={args.tuned} total_timesteps={args.total_timesteps} "
          f"device={args.device}")
    if tb_log:
        print(f"[sb3_recoco_reproduction] tensorboardログ: {tb_log}/{run_name}_1 "
              f"(別ターミナルで 'tensorboard --logdir {tb_log} --host 0.0.0.0' "
              f"を実行すると学習曲線をブラウザで確認できます)")

    model.learn(total_timesteps=args.total_timesteps, progress_bar=False,
                tb_log_name=run_name)
    model.save(args.output)
    print(f"[sb3_recoco_reproduction] 保存しました: {args.output}")

    if args.eval_episodes > 0:
        print(f"[sb3_recoco_reproduction] 学習後の評価を実行します"
              f"(trace毎に{args.eval_episodes}エピソード)...")
        results = run_evaluation(model, args.mode, args.trace, args.trace_set,
                                  n_episodes=args.eval_episodes)
        save_eval_csv(results, args.output)
        overall_mean = float(np.mean([r["mean_reward"] for r in results if r["mean_reward"] is not None]))
        print(f"[sb3_recoco_reproduction] 全trace平均reward: {overall_mean:.4f}")


if __name__ == "__main__":
    main()
