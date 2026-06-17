# prepare_Command_TimeDataset.py

import argparse
import shutil
from pathlib import Path
import json
import pandas as pd

from prepare_Command_Dataset import (
    read_command_json_to_tsv,
    build_users_and_behaviors,
    save_tsv_no_header,
)

def merge_viewlogs(train_viewlog_json: Path, test_viewlog_json: Path, out_viewlog_json: Path):
    with open(train_viewlog_json, "r", encoding="utf-8") as f:
        train_logs = json.load(f)

    with open(test_viewlog_json, "r", encoding="utf-8") as f:
        test_logs = json.load(f)

    all_logs = train_logs + test_logs

    with open(out_viewlog_json, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, ensure_ascii=False)

    print(f"[viewlog_all.json] saved: {out_viewlog_json} rows={len(all_logs)}")


def split_behaviors_by_time_bins(beh_df: pd.DataFrame, bin_num: int = 10):
    beh_df = beh_df.copy()

    beh_df["ReportTime_dt"] = pd.to_datetime(
        beh_df["ReportTime"],
        errors="coerce"
    )

    beh_df = beh_df.sort_values(
        "ReportTime_dt",
        na_position="last"
    ).reset_index(drop=True)

    n = len(beh_df)
    if n == 0:
        raise ValueError("[time split] behaviors.tsv is empty")

    if bin_num < 2:
        raise ValueError("[time split] bin_num must be >= 2")

    bin_size = n // bin_num
    if bin_size == 0:
        raise ValueError(
            f"[time split] data too small: rows={n}, bin_num={bin_num}"
        )

    bins = []
    for i in range(bin_num):
        start = i * bin_size
        end = n if i == bin_num - 1 else (i + 1) * bin_size

        bin_df = (
            beh_df.iloc[start:end]
            .drop(columns=["ReportTime_dt"])
            .reset_index(drop=True)
        )

        bins.append(bin_df)
        print(f"[time bin {i + 1}] rows={len(bin_df)}")

    return bins


def copy_static_files(command_tsv: Path, users_tsv: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(command_tsv, target_dir / "commands.tsv")
    shutil.copy2(users_tsv, target_dir / "users.tsv")


def merge_command_tsv(train_command_tsv: Path, test_command_tsv: Path, out_command_tsv: Path):
    train_df = pd.read_csv(train_command_tsv, sep="\t", header=None, dtype=str)
    test_df = pd.read_csv(test_command_tsv, sep="\t", header=None, dtype=str)

    all_cmd_df = pd.concat([train_df, test_df], ignore_index=True)

    # dataId 기준 중복 제거
    all_cmd_df = all_cmd_df.drop_duplicates(subset=[0], keep="first").reset_index(drop=True)

    out_command_tsv.parent.mkdir(parents=True, exist_ok=True)
    all_cmd_df.to_csv(
        out_command_tsv,
        sep="\t",
        header=False,
        index=False,
        encoding="utf-8",
        lineterminator="\n"
    )

    print(f"[command_all.tsv] saved: {out_command_tsv} rows={len(all_cmd_df)}")


def prepare_command_time_dataset(
    *,
    out_dir: str = "../Command-April/time",
    train_command_json: str = "data/command_samples_train_2604.json",
    train_viewlog_json: str = "data/user_command_viewlog_train_2604.json",
    test_command_json: str = "data/command_samples_test_2604.json",
    test_viewlog_json: str = "data/user_command_viewlog_test_2604.json",
    users_json: str = "data/users_samples_260103.json",
    hanhwa_csv: str = "data/hanhwa_report.csv",
    bin_num: int = 10,
):
    out_dir = Path(out_dir)
    gen_dir = out_dir / "_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    users_json = Path(users_json)

    train_command_tsv = gen_dir / "command_train.tsv"
    train_behaviors_tsv = gen_dir / "behaviors_train.tsv"

    test_command_tsv = gen_dir / "command_test.tsv"
    test_behaviors_tsv = gen_dir / "behaviors_test.tsv"

    all_command_tsv = gen_dir / "commands_all.tsv"
    all_behaviors_tsv = gen_dir / "behaviors_all.tsv"
    out_users = gen_dir / "users.tsv"

    all_viewlog_json = gen_dir / "viewlog_all.json"

    # 1. train/test command 생성
    read_command_json_to_tsv(
        train_command_json,
        hanhwa_csv,
        str(train_command_tsv)
    )

    read_command_json_to_tsv(
        test_command_json,
        hanhwa_csv,
        str(test_command_tsv)
    )

    # 2. command 합치기
    merge_command_tsv(
        train_command_tsv=train_command_tsv,
        test_command_tsv=test_command_tsv,
        out_command_tsv=all_command_tsv,
    )

    # 3. viewlog 합치기
    merge_viewlogs(
        Path(train_viewlog_json),
        Path(test_viewlog_json),
        all_viewlog_json,
    )

    # 4. 전체 기준 behavior/users 생성 (딱 1번)
    all_beh_df, _ = build_users_and_behaviors(
        command_tsv=all_command_tsv,
        viewlog_json=all_viewlog_json,
        users_json=users_json,
        out_behaviors=all_behaviors_tsv,
        out_users=out_users,
    )

 
    # ImpressionID 재부여
    all_beh_df = all_beh_df.reset_index(drop=True)
    all_beh_df["ImpressionID"] = range(1, len(all_beh_df) + 1)

    save_tsv_no_header(all_beh_df, all_behaviors_tsv)

    print(f"[behaviors_all.tsv] rows={len(all_beh_df)}")

    # 5. 전체 데이터를 ReportTime 기준으로 10구간 분할
    bins = split_behaviors_by_time_bins(
        all_beh_df,
        bin_num=bin_num
    )

    # 6. train_1 ~ train_9 생성
    for i in range(1, bin_num):
        train_df = pd.concat(bins[:i], ignore_index=True)

        # ImpressionID 재부여
        train_df = train_df.reset_index(drop=True)
        train_df["ImpressionID"] = range(1, len(train_df) + 1)

        train_dir = out_dir / f"train_{i}"
        save_tsv_no_header(train_df, train_dir / "behaviors.tsv")
        copy_static_files(all_command_tsv, out_users, train_dir)

        print(f"[time train_{i}] bins=1~{i}, rows={len(train_df)}")

    # 7. 마지막 10구간을 test로 저장
    test_df = bins[-1].reset_index(drop=True)
    test_df["ImpressionID"] = range(1, len(test_df) + 1)

    test_dir = out_dir / "test"
    save_tsv_no_header(test_df, test_dir / "behaviors.tsv")
    copy_static_files(all_command_tsv, out_users, test_dir)

    # dev도 test와 동일하게 생성
    dev_dir = out_dir / "dev"
    save_tsv_no_header(test_df, dev_dir / "behaviors.tsv")
    copy_static_files(all_command_tsv, out_users, dev_dir)

    print("\n[DONE] temporal dataset completed.")
    print("  out_dir:", out_dir.resolve())
    print(f"  train folders: train_1 ~ train_{bin_num - 1}")
    print("  dev folder    : dev")
    print("  test folder   : test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", default="../Command-April/time")

    parser.add_argument("--train_command_json", default="data/command_samples_train_2604.json")
    parser.add_argument("--train_viewlog_json", default="data/user_command_viewlog_train_2604.json")

    parser.add_argument("--test_command_json", default="data/command_samples_test_2604.json")
    parser.add_argument("--test_viewlog_json", default="data/user_command_viewlog_test_2604.json")

    parser.add_argument("--users_json", default="data/users_samples_260103.json")
    parser.add_argument("--hanhwa_csv", default="data/hanhwa_report.csv")
    parser.add_argument("--bin_num", type=int, default=10)

    args = parser.parse_args()
    prepare_command_time_dataset(**vars(args))