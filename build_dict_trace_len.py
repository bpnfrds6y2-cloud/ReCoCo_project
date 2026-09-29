
import argparse
import glob
import json
import os
import pickle
 
 
def compute_duration_sec(json_path):
    """ReCoCo/OpenNetLab形式のトレースJSON
    ( {"uplink": {"trace_pattern": [{"capacity": kbps, "duration": ms}, ...]}} )
    を読み込み、全セグメントの duration(ms)を合計して秒に変換する。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    if isinstance(data, dict) and isinstance(data.get("uplink"), dict) \
            and "trace_pattern" in data["uplink"]:
        pattern = data["uplink"]["trace_pattern"]
        total_ms = sum(float(seg.get("duration", 0.0)) for seg in pattern)
        return total_ms / 1000.0
 
    raise ValueError(
        f"{json_path}: 'uplink'->'trace_pattern' 形式のトレースではありません。"
        "ReCoCo/OpenNetLab形式のJSONを指定してください。"
    )
 
 
def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces-dir", type=str, default="./traces",
                   help="トレースJSONファイルが入っているディレクトリ(デフォルト: ./traces)")
    p.add_argument("--pattern", type=str, default="*.json",
                   help="トレースファイルのglobパターン(デフォルト: *.json)")
    p.add_argument("--output", type=str, default="./dict_trace_len.pickle",
                   help="出力するpickleファイルのパス(デフォルト: ./dict_trace_len.pickle、"
                        "rtc_env.pyと同じディレクトリに置くこと)")
    p.add_argument("--print-only", action="store_true",
                   help="pickleに保存せず、生成される辞書の内容を表示するだけ")
    return p
 
 
def main():
    args = build_arg_parser().parse_args()
 
    search_pattern = os.path.join(args.traces_dir, args.pattern)
    trace_paths = sorted(glob.glob(search_pattern))
 
    if not trace_paths:
        raise SystemExit(
            f"トレースファイルが見つかりませんでした: {search_pattern}\n"
            "--traces-dir でトレースの入っているフォルダを指定してください。"
        )
 
    dict_trace_len = {}
    n_ok, n_fail = 0, 0
    for path in trace_paths:
        try:
            duration_sec = compute_duration_sec(path)
            dict_trace_len[path] = {"duration": duration_sec}
            print(f"[ok]   {path}: duration = {duration_sec:.2f} sec "
                  f"({duration_sec/60:.2f} min)")
            n_ok += 1
        except Exception as e:
            print(f"[skip] {path}: 読み込み失敗 ({e})")
            n_fail += 1
 
    if not dict_trace_len:
        raise SystemExit("有効なトレースが1つもありませんでした。処理を中止します。")
 
    print(f"\n{n_ok}本のトレースを登録しました(失敗: {n_fail}本)。")
 
    if args.print_only:
        print("\n--print-only が指定されたため、pickleへの保存はスキップしました。")
        print("生成される辞書のキー一覧:")
        for k in dict_trace_len:
            print(f"  {k}")
        return
 
    with open(args.output, "wb") as f:
        pickle.dump(dict_trace_len, f)
    print(f"\n{args.output} に保存しました。")
    print(
        "\n[重要] run_experiments_all_in_one.py 側の --global-traces / "
        "--client-traces には、ここで使ったのと同じ書き方のパス"
        f"(例: {trace_paths[0]!r} のように先頭の './' も含めて)を"
        "指定してください。書き方が1文字でも違うと KeyError になります。"
    )
 
 
if __name__ == "__main__":
    main()