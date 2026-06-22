# prepare_Command_Dataset.py
import argparse
import csv
import json
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# =========================================================
# Config
# =========================================================
KST = timezone(timedelta(hours=9))

CMD_COLS = ["dataId", "validUntil", "securityLevel", "title", "reportTime", "body", "category"]

# users.tsv (사용자 정보 - 현재 유지: 헤더 없음)
# columns: userId, name, department, position, rank, unit, history
USER_COLS = ["UserID", "name", "department", "position", "rank", "unit", "History"]

# behaviors.tsv (USER 추천으로 변경: 헤더 없음)
# columns: ImpressionID, CommandID, ReportTime, Impressions (명령과 어울리는 사용자 추천)
# Impressions 형식: userId1-label1 userId2-label2 ... (label=1 이면 해당 user가 명령을 읽음)
BEH_COLS = ["ImpressionID", "CommandID", "ReportTime", "Impressions"]


# =========================================================
# Utils
# =========================================================
def to_iso_kst(s: str) -> str:
    """다양한 날짜 문자열을 KST ISO8601로 최대한 정규화. 파싱 실패 시 원문 반환."""
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    # 1) ISO-ish
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        else:
            dt = dt.astimezone(KST)
        return dt.isoformat(timespec="seconds")
    except Exception:
        pass

    # 2) '2025년 11월 14일 13시 42분' etc.
    m = re.search(
        r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*(\d{1,2})\s*시\s*(\d{1,2})\s*분(?:\s*(\d{1,2})\s*초)?",
        s,
    )
    if m:
        y, mo, d, h, mi, sec = m.groups()
        sec = sec or "0"
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(sec), tzinfo=KST)
        return dt.isoformat(timespec="seconds")

    return s


def sanitize_cell(x: str) -> str:
    """TSV 깨짐 방지: 탭/개행 제거."""
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return s


# =========================================================
# 1) command.tsv 생성 (기존 유지: json + csv(2nd col time))
# =========================================================
'''
def read_command_json_to_tsv(json_path: str, csv_path: str, out_tsv: str) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    # csv 2nd col as time
    times: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            times.append(row[1] if len(row) >= 2 else "")

    # drop header-like
    if times and isinstance(times[0], str) and ("time" in times[0].lower() or "report" in times[0].lower()):
        times = times[1:]

    n_json = len(records)
    if len(times) < n_json:
        times += [""] * (n_json - len(times))
    else:
        times = times[:n_json]

    times = [to_iso_kst(t) for t in times]

    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for r, t in zip(records, times):
            row = [
                sanitize_cell(r.get("dataId", "")),
                sanitize_cell(r.get("validUntil", "")),
                sanitize_cell(r.get("securityLevel", "")),
                sanitize_cell(r.get("title", "")),
                sanitize_cell(t),
                sanitize_cell(r.get("body", "")),
                sanitize_cell(r.get("category", "")),
            ]
            w.writerow(row)

    print(f"[command.tsv] saved: {out_tsv} (n={n_json})")
'''
def read_command_json_to_tsv(json_path: str, csv_path: str, out_tsv: str) -> None:
    json_path = Path(json_path)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    n_json = len(records)

    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")

        for r in records:

            # reportTime을 updatedAt에서 가져오기
            report_time = r.get("reportTime") or r.get("updatedAt", "")
            report_time = to_iso_kst(report_time)

            row = [
                sanitize_cell(r.get("dataId", "")),
                sanitize_cell(r.get("validUntil", "")),
                sanitize_cell(r.get("securityLevel", "")),
                sanitize_cell(r.get("title", "")),
                sanitize_cell(report_time),
                sanitize_cell(r.get("body", "")),
                sanitize_cell(r.get("category", "")),
            ]

            w.writerow(row)

    print(f"[command.tsv] saved: {out_tsv} (n={n_json})")

def load_command_tsv(command_tsv: Path) -> pd.DataFrame:
    cmd_df = pd.read_csv(command_tsv, sep="\t", header=None, names=CMD_COLS, dtype=str)
    cmd_df["dataId"] = cmd_df["dataId"].astype(str).str.strip()
    cmd_df["reportTime"] = cmd_df["reportTime"].astype(str).str.strip()
    return cmd_df


# =========================================================
# 2) users.tsv + behaviors.tsv 생성 (User Recommendation 방식)
# =========================================================
def build_users_and_behaviors(
    *,
    command_tsv: Path,
    viewlog_json: Path,
    users_json: Path,
    out_behaviors: Path,
    out_users: Path,
    # 수정
    history_viewlog_json: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cmd_df = load_command_tsv(command_tsv)
    dataid_to_time: Dict[str, str] = dict(zip(cmd_df["dataId"], cmd_df["reportTime"]))

    # users.json 로드
    with users_json.open("r", encoding="utf-8") as f:
        users_data = json.load(f)

    users = users_data.get("users", [])
    if not isinstance(users, list):
        raise ValueError("USERS_JSON의 최상위 'users'가 list가 아닙니다.")

    # 순서 고정(중요)
    all_user_ids = [str(u.get("userId", "")).strip() for u in users]
    all_user_ids = [uid for uid in all_user_ids if uid]
    
    '''
    # viewlog 로드
    with viewlog_json.open("r", encoding="utf-8") as f:
        viewlog = json.load(f)
    if not isinstance(viewlog, list):
        raise ValueError("VIEWLOG_JSON은 list[{dataId, userIds}] 형식이어야 합니다.")
    '''
    # 수정
    # behaviors 생성용 viewlog 로드
    with viewlog_json.open("r", encoding="utf-8") as f:
        viewlog = json.load(f)

    if not isinstance(viewlog, list):
        raise ValueError("VIEWLOG_JSON은 list[{dataId, userIds}] 형식이어야 합니다.")

    # history 생성용 viewlog 로드
    if history_viewlog_json is None:
        history_viewlog = viewlog
    else:
        with history_viewlog_json.open("r", encoding="utf-8") as f:
            history_viewlog = json.load(f)

    if not isinstance(history_viewlog, list):
        raise ValueError("history_viewlog_json은 list[{dataId, userIds}] 형식이어야 합니다.")

    # command별 readers set + user별 history 누적
    cmd_to_readers: Dict[str, Set[str]] = {}
    user_to_history: Dict[str, List[str]] = {uid: [] for uid in all_user_ids}

    # 수정
    #for row in viewlog:
    for row in history_viewlog:
        did = str(row.get("dataId", "")).strip()
        readers = [str(x).strip() for x in (row.get("userIds", []) or []) if str(x).strip()]
        read_set = set(readers)

        if did:
            cmd_to_readers[did] = read_set

        for uid in read_set:
            if uid in user_to_history and did:
                user_to_history[uid].append(did)

    # 4) behaviors.tsv 생성 (USER 추천 방식: command 중심)
    # 각 행은 하나의 command를 나타내고, 각 command에 대해 어떤 user가 읽었는지 표현
    out_rows = []
    for cmd_idx, did in enumerate(cmd_df["dataId"], start=1):
        did_str = str(did).strip()
        report_time = dataid_to_time.get(did_str, "")
        report_time = sanitize_cell(report_time)

        read_set = cmd_to_readers.get(did_str, set())

        # 각 user에 대해 label 표시 (1: 해당 user가 명령을 읽음, 0: 읽지 않음)
        tokens = []
        for uid in all_user_ids:
            label = "1" if uid in read_set else "0"
            tokens.append(f"{uid}-{label}")

        impressions = " ".join(tokens)
        out_rows.append([cmd_idx, did_str, report_time, impressions])

    beh_df = pd.DataFrame(out_rows, columns=BEH_COLS)
    out_behaviors.parent.mkdir(parents=True, exist_ok=True)
    beh_df.to_csv(out_behaviors, sep="\t", header=False, index=False, encoding="utf-8", lineterminator="\n")

    # 5) users.tsv 생성 (헤더 없음)
    user_rows = []
    for u in users:
        uid = str(u.get("userId", "")).strip()
        if not uid:
            continue
        history = " ".join(user_to_history.get(uid, []))

        user_rows.append(
            [
                uid,
                sanitize_cell(u.get("name", "")),
                sanitize_cell(u.get("department", "")),
                sanitize_cell(u.get("position", "")),
                sanitize_cell(u.get("rank", "")),
                sanitize_cell(u.get("unit", "")),
                sanitize_cell(history),
            ]
        )

    user_df = pd.DataFrame(user_rows, columns=USER_COLS)
    out_users.parent.mkdir(parents=True, exist_ok=True)
    user_df.to_csv(out_users, sep="\t", header=False, index=False, encoding="utf-8", lineterminator="\n")

    print(f"[behaviors.tsv] saved: {out_behaviors} (rows={len(beh_df)} users)")
    print(f"[users.tsv]     saved: {out_users} (rows={len(user_df)})")
    return beh_df, user_df


# =========================================================
# 3) behaviors.tsv split + command/users 복사
# =========================================================
def split_behaviors_random(
    beh_df: pd.DataFrame,
    train_ratio: float,
    dev_ratio_in_train: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    idx = list(beh_df.index)
    rng.shuffle(idx)

    n = len(idx)
    n_test = int(round(n * (1.0 - train_ratio)))
    test_idx = idx[:n_test]
    train_idx_full = idx[n_test:]

    train_full = beh_df.loc[train_idx_full].reset_index(drop=True)
    test = beh_df.loc[test_idx].reset_index(drop=True)

    idx2 = list(train_full.index)
    rng.shuffle(idx2)
    n_dev = max(1, int(round(len(idx2) * dev_ratio_in_train))) if len(idx2) > 1 else 0
    dev_idx = idx2[:n_dev]
    train_idx = idx2[n_dev:]

    train = train_full.loc[train_idx].reset_index(drop=True)
    dev = train_full.loc[dev_idx].reset_index(drop=True)

    print("[split behaviors] random")
    print("  train:", len(train), "dev:", len(dev), "test:", len(test))
    return train, dev, test

def split_behaviors_train_dev_only(
    beh_df: pd.DataFrame,
    dev_ratio_in_train: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    idx = list(beh_df.index)
    rng.shuffle(idx)

    n_dev = max(1, int(round(len(idx) * dev_ratio_in_train))) if len(idx) > 1 else 0
    dev_idx = idx[:n_dev]
    train_idx = idx[n_dev:]

    train = beh_df.loc[train_idx].reset_index(drop=True)
    dev = beh_df.loc[dev_idx].reset_index(drop=True)

    print("[split behaviors] train/dev only")
    print("  train:", len(train), "dev:", len(dev))
    return train, dev


def save_tsv_no_header(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", header=False, index=False, encoding="utf-8", lineterminator="\n")
    print("[save]", path, f"(rows={len(df)})")


def copy_static_files_to_splits(
    *,
    command_tsv: Path,
    users_tsv: Path,
    out_dir: Path,
) -> None:
    train_dir = out_dir / "train"
    dev_dir = out_dir / "dev"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for d in [train_dir, dev_dir, test_dir]:
        shutil.copy2(command_tsv, d / "commands.tsv")  # MIND 관례: commands.tsv
        shutil.copy2(users_tsv, d / "users.tsv")

    print("[copy] commands.tsv & users.tsv -> train/dev/test")


# =========================================================
# Main Pipeline
# =========================================================
def prepare_command_dataset(
    *,
    out_dir: str = "../Command-March",
    command_json: str = "data/command_samples_train_2605.json",
    hanhwa_csv: str = "data/hanhwa_report.csv",
    command_tsv: str = "data/command.tsv",
    viewlog_json: str = "data/user_command_viewlog_train_2605.json",
    test_command_json: str = "data/command_samples_test_2605.json",
    test_viewlog_json: str = "data/user_command_viewlog_test_2605.json",
    users_json: str = "data/users_samples_260103.json",
    seed: int = 42,
    train_ratio: float = 0.90,
    dev_ratio_in_train: float = 0.05,
):

    command_tsv = Path(command_tsv)
    out_dir = Path(out_dir)
    gen_dir = out_dir / "_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    users_json = Path(users_json)
    viewlog_json = Path(viewlog_json)

    test_command_json = Path(test_command_json) if test_command_json else None
    test_viewlog_json = Path(test_viewlog_json) if test_viewlog_json else None

    use_external_test = (
        test_command_json is not None
        and test_viewlog_json is not None
        and test_command_json.exists()
        and test_viewlog_json.exists()
    )

    # (1) train 원본 command.tsv 생성
    read_command_json_to_tsv(command_json, hanhwa_csv, str(command_tsv))

    # (2) train 원본 behaviors/users 생성
    out_behaviors = gen_dir / "behaviors.tsv"
    out_users = gen_dir / "users.tsv"

    beh_df, _user_df = build_users_and_behaviors(
        command_tsv=command_tsv,
        viewlog_json=viewlog_json,
        users_json=users_json,
        out_behaviors=out_behaviors,
        out_users=out_users,
    )

    # -----------------------------
    # A. test 파일이 있으면: train/dev만 나누기
    # -----------------------------
    if use_external_test:
        train_df, dev_df = split_behaviors_train_dev_only(
            beh_df=beh_df,
            dev_ratio_in_train=dev_ratio_in_train,
            seed=seed,
        )

        save_tsv_no_header(train_df, out_dir / "train" / "behaviors.tsv")
        save_tsv_no_header(dev_df, out_dir / "dev" / "behaviors.tsv")

        # train/dev에는 train command 복사
        (out_dir / "train").mkdir(parents=True, exist_ok=True)
        (out_dir / "dev").mkdir(parents=True, exist_ok=True)
        shutil.copy2(command_tsv, out_dir / "train" / "commands.tsv")
        shutil.copy2(command_tsv, out_dir / "dev" / "commands.tsv")
        shutil.copy2(out_users, out_dir / "train" / "users.tsv")
        shutil.copy2(out_users, out_dir / "dev" / "users.tsv")

        # test는 외부 test 파일 사용
        test_command_tsv = gen_dir / "command_test.tsv"
        test_behaviors_tsv = gen_dir / "behaviors_test.tsv"

        read_command_json_to_tsv(str(test_command_json), hanhwa_csv, str(test_command_tsv))

        '''
        test_df, _ = build_users_and_behaviors(
            command_tsv=test_command_tsv,
            viewlog_json=test_viewlog_json,
            users_json=users_json,
            out_behaviors=test_behaviors_tsv,
            out_users=out_users,
        )
        '''
        # 수정: history_viewlog_json 추가
        test_df, _ = build_users_and_behaviors(
            command_tsv=test_command_tsv,
            viewlog_json=test_viewlog_json,      # test behavior / label
            history_viewlog_json=viewlog_json,   # train history 사용
            users_json=users_json,
            out_behaviors=test_behaviors_tsv,
            out_users=out_users,
        )

        save_tsv_no_header(test_df, out_dir / "test" / "behaviors.tsv")
        (out_dir / "test").mkdir(parents=True, exist_ok=True)
        shutil.copy2(test_command_tsv, out_dir / "test" / "commands.tsv")
        shutil.copy2(out_users, out_dir / "test" / "users.tsv")

        print("[test] external test used")
        print("       train source -> train/dev")
        print("       external test source -> test")

    # -----------------------------
    # B. test 파일이 없으면: 기존처럼 train/dev/test 나누기
    # -----------------------------
    else:
        train_df, dev_df, test_df = split_behaviors_random(
            beh_df=beh_df,
            train_ratio=train_ratio,
            dev_ratio_in_train=dev_ratio_in_train,
            seed=seed,
        )

        save_tsv_no_header(train_df, out_dir / "train" / "behaviors.tsv")
        save_tsv_no_header(dev_df, out_dir / "dev" / "behaviors.tsv")
        save_tsv_no_header(test_df, out_dir / "test" / "behaviors.tsv")

        # 기존처럼 전부 같은 command/users 복사
        for split in ["train", "dev", "test"]:
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(command_tsv, split_dir / "commands.tsv")
            shutil.copy2(out_users, split_dir / "users.tsv")

        print("[test] no external test -> random train/dev/test split used")

    print("\n[DONE] behaviors split + static copy completed.")
    print("  out_dir:", out_dir.resolve())
    print("  generated:", gen_dir.resolve())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # command.tsv 생성 입력
    ap.add_argument("--command_json", default="data/command_samples_train_2605.json")
    ap.add_argument("--hanhwa_csv", default="data/hanhwa_report.csv")
    ap.add_argument("--command_tsv", default="data/command.tsv")

    # user/behavior 생성 입력
    ap.add_argument("--viewlog_json", default="data/user_command_viewlog_train_2605.json")
    ap.add_argument("--test_command_json", default="data/command_samples_test_2605.json")
    ap.add_argument("--test_viewlog_json", default="data/user_command_viewlog_test_2605.json")
    ap.add_argument("--users_json", default="data/user_sample_260103.json")

    # 출력
    ap.add_argument("--out_dir", default="../Command-May")

    # split 비율
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.90)        # train=90%
    ap.add_argument("--dev_ratio_in_train", type=float, default=0.05) # train의 5%를 dev로

    args = ap.parse_args()

    prepare_command_dataset(**vars(args))
