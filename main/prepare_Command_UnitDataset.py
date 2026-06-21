# prepare_Command_UnitDataset.py
import argparse
import csv
import json
import random
import re
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

KST = timezone(timedelta(hours=9))

CMD_COLS = ["dataId", "validUntil", "securityLevel", "title", "reportTime", "body", "category"]

# units.tsv
# columns: unitId, name, unitSize, unitType, combatPower, location, history
UNIT_COLS = ["UnitID", "name", "unitSize", "unitType", "combatPower", "location", "History"]

# behaviors.tsv
# columns: ImpressionID, CommandID, ReportTime, Impressions
# Impressions: unit-001-1 unit-002-0 ...
BEH_COLS = ["ImpressionID", "CommandID", "ReportTime", "Impressions"]


def to_iso_kst(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        else:
            dt = dt.astimezone(KST)
        return dt.isoformat(timespec="seconds")
    except Exception:
        pass

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


def sanitize_cell(x) -> str:
    if x is None:
        return ""
    return str(x).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def read_command_json_to_tsv(json_path: str, out_tsv: str) -> None:
    json_path = Path(json_path)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")

        for r in records:
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

    print(f"[commands.tsv] saved: {out_tsv} (n={len(records)})")


def load_command_tsv(command_tsv: Path) -> pd.DataFrame:
    cmd_df = pd.read_csv(command_tsv, sep="\t", header=None, names=CMD_COLS, dtype=str)
    cmd_df["dataId"] = cmd_df["dataId"].astype(str).str.strip()
    cmd_df["reportTime"] = cmd_df["reportTime"].astype(str).str.strip()
    return cmd_df


def build_units_and_behaviors(
    *,
    command_tsv: Path,
    viewlog_json: Path,
    units_json: Path,
    out_behaviors: Path,
    out_units: Path,
    # 히스토리 설정
    history_viewlog_json: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    cmd_df = load_command_tsv(command_tsv)
    dataid_to_time: Dict[str, str] = dict(zip(cmd_df["dataId"], cmd_df["reportTime"]))

    with units_json.open("r", encoding="utf-8") as f:
        units_data = json.load(f)

    units = units_data.get("units", [])
    if not isinstance(units, list):
        raise ValueError("units_json의 최상위 'units'가 list가 아닙니다.")

    all_unit_ids = [str(u.get("unitId", "")).strip() for u in units]
    all_unit_ids = [uid for uid in all_unit_ids if uid]

    # behaviors 생성용
    with viewlog_json.open("r", encoding="utf-8") as f:
        viewlog = json.load(f)

    # history 생성용
    if history_viewlog_json is None:
        history_viewlog = viewlog
    else:
        with history_viewlog_json.open("r", encoding="utf-8") as f:
            history_viewlog = json.load(f)

    if not isinstance(viewlog, list):
        raise ValueError("viewlog_json은 list[{dataId, unitIds}] 형식이어야 합니다.")

    cmd_to_units: Dict[str, Set[str]] = {}
    unit_to_history: Dict[str, List[str]] = {uid: [] for uid in all_unit_ids}

    # 원래는 behaviors와 동일한 viewlog로 history 생성
    # for row in viewlog:

    # 수정 : history_viewlog 사용
    for row in history_viewlog:
        did = str(row.get("dataId", "")).strip()
        unit_ids = [
            str(x).strip()
            for x in (row.get("unitIds", []) or [])
            if str(x).strip()
        ]

        read_set = set(unit_ids)

        if did:
            cmd_to_units[did] = read_set

        for uid in read_set:
            if uid in unit_to_history and did:
                unit_to_history[uid].append(did)

    # behaviors.tsv 생성
    out_rows = []

    for cmd_idx, did in enumerate(cmd_df["dataId"], start=1):
        did_str = str(did).strip()
        report_time = sanitize_cell(dataid_to_time.get(did_str, ""))

        read_set = cmd_to_units.get(did_str, set())

        tokens = []
        for uid in all_unit_ids:
            label = "1" if uid in read_set else "0"
            tokens.append(f"{uid}-{label}")

        impressions = " ".join(tokens)
        out_rows.append([cmd_idx, did_str, report_time, impressions])

    beh_df = pd.DataFrame(out_rows, columns=BEH_COLS)
    out_behaviors.parent.mkdir(parents=True, exist_ok=True)
    beh_df.to_csv(
        out_behaviors,
        sep="\t",
        header=False,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    # units.tsv 생성
    unit_rows = []

    for u in units:
        uid = str(u.get("unitId", "")).strip()
        if not uid:
            continue

        history = " ".join(unit_to_history.get(uid, []))

        unit_rows.append(
            [
                uid,
                sanitize_cell(u.get("name", "")),
                sanitize_cell(u.get("unitSize", "")),
                sanitize_cell(u.get("unitType", "")),
                sanitize_cell(u.get("combatPower", "")),
                sanitize_cell(u.get("location", "")),
                sanitize_cell(history),
            ]
        )

    unit_df = pd.DataFrame(unit_rows, columns=UNIT_COLS)
    out_units.parent.mkdir(parents=True, exist_ok=True)
    unit_df.to_csv(
        out_units,
        sep="\t",
        header=False,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    print(f"[behaviors.tsv] saved: {out_behaviors} (rows={len(beh_df)})")
    print(f"[units.tsv]     saved: {out_units} (rows={len(unit_df)})")

    return beh_df, unit_df


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
    df.to_csv(
        path,
        sep="\t",
        header=False,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    print("[save]", path, f"(rows={len(df)})")


def prepare_command_unit_dataset(
    *,
    out_dir: str = "../Command-unit",
    command_json: str = "data/unit/command_samples_train_2605.json",
    command_tsv: str = "data/unit/command.tsv",
    viewlog_json: str = "data/unit/unit_command_viewlog_train_2605.json",
    test_command_json: str = "data/unit/command_samples_val_2605.json",
    test_viewlog_json: str = "data/unit/unit_command_viewlog_val_2605.json",
    units_json: str = "data/unit/units_samples_251110.json",
    seed: int = 42,
    dev_ratio_in_train: float = 0.05,
):

    out_dir = Path(out_dir)
    gen_dir = out_dir / "_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    command_tsv = Path(command_tsv)
    viewlog_json = Path(viewlog_json)
    units_json = Path(units_json)

    test_command_json = Path(test_command_json)
    test_viewlog_json = Path(test_viewlog_json)

    use_external_test = test_command_json.exists() and test_viewlog_json.exists()

    # train command 생성
    read_command_json_to_tsv(command_json, str(command_tsv))

    # train behaviors / units 생성
    out_behaviors = gen_dir / "behaviors.tsv"
    out_units = gen_dir / "units.tsv"

    beh_df, _unit_df = build_units_and_behaviors(
        command_tsv=command_tsv,
        viewlog_json=viewlog_json,
        units_json=units_json,
        out_behaviors=out_behaviors,
        out_units=out_units,
    )

    # train/dev split
    train_df, dev_df = split_behaviors_train_dev_only(
        beh_df=beh_df,
        dev_ratio_in_train=dev_ratio_in_train,
        seed=seed,
    )

    save_tsv_no_header(train_df, out_dir / "train" / "behaviors.tsv")
    save_tsv_no_header(dev_df, out_dir / "dev" / "behaviors.tsv")

    for split in ["train", "dev"]:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(command_tsv, split_dir / "commands.tsv")
        shutil.copy2(out_units, split_dir / "units.tsv")

    # external val/test 사용
    if use_external_test:
        test_command_tsv = gen_dir / "command_test.tsv"
        test_behaviors_tsv = gen_dir / "behaviors_test.tsv"

        read_command_json_to_tsv(str(test_command_json), str(test_command_tsv))

        '''
        test_df, _ = build_units_and_behaviors(
            command_tsv=test_command_tsv,
            viewlog_json=test_viewlog_json,
            units_json=units_json,
            out_behaviors=test_behaviors_tsv,
            out_units=out_units,
        )
        '''

        test_df, _ = build_units_and_behaviors(
            command_tsv=test_command_tsv,
            viewlog_json=test_viewlog_json,          # test behavior
            history_viewlog_json=viewlog_json,       # train history 사용
            units_json=units_json,
            out_behaviors=test_behaviors_tsv,
            out_units=out_units,
        )

        save_tsv_no_header(test_df, out_dir / "test" / "behaviors.tsv")

        test_dir = out_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(test_command_tsv, test_dir / "commands.tsv")
        shutil.copy2(out_units, test_dir / "units.tsv")

        print("[test] external val/test used")
    else:
        # test 파일 없으면 dev를 test로 복사
        save_tsv_no_header(dev_df, out_dir / "test" / "behaviors.tsv")
        test_dir = out_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(command_tsv, test_dir / "commands.tsv")
        shutil.copy2(out_units, test_dir / "units.tsv")

        print("[test] no external test -> dev copied to test")

    print("\n[DONE] unit dataset completed.")
    print("  out_dir:", out_dir.resolve())
    print("  generated:", gen_dir.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", default="../Command-unit")

    ap.add_argument("--command_json", default="data/unit/command_samples_train_2605.json")
    ap.add_argument("--command_tsv", default="data/unit/command.tsv")
    ap.add_argument("--viewlog_json", default="data/unit/unit_command_viewlog_train_2605.json")

    ap.add_argument("--test_command_json", default="data/unit/command_samples_val_2605.json")
    ap.add_argument("--test_viewlog_json", default="data/unit/unit_command_viewlog_val_2605.json")

    ap.add_argument("--units_json", default="data/unit/units_samples_251110.json")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev_ratio_in_train", type=float, default=0.05)

    args = ap.parse_args()

    prepare_command_unit_dataset(**vars(args))