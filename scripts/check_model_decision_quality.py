from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from quant_platform.services.settings import load_settings


PANEL_REQUIRED_COLS = {
    "evaluation_version",
    "instrument_id",
    "symbol",
    "split_id",
    "dataset_role",
    "metric_name",
    "benchmark_metric_value",
    "ml_metric_value",
    "optimization_direction",
    "ml_minus_benchmark",
    "benchmark_minus_ml",
    "relative_improvement_vs_benchmark",
}

SUMMARY_REQUIRED_COLS = {
    "evaluation_version",
    "instrument_id",
    "symbol",
    "benchmark_model_name",
    "ml_model_name",
    "benchmark_mean_qlike",
    "ml_mean_qlike",
    "relative_qlike_improvement_mean",
    "benchmark_mean_macro_f1",
    "ml_mean_macro_f1",
    "macro_f1_delta",
    "benchmark_mean_balanced_accuracy",
    "ml_mean_balanced_accuracy",
    "balanced_accuracy_delta",
    "qlike_pass",
    "macro_f1_pass",
    "balanced_accuracy_pass",
    "calibration_status",
    "decision",
    "comparison_rows",
    "distinct_splits",
    "distinct_roles",
}

REASONS_REQUIRED_COLS = {
    "evaluation_version",
    "instrument_id",
    "symbol",
    "decision",
    "rule_name",
    "expected_threshold",
    "observed_value",
    "passed",
}


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: set[str],
    df_name: str,
    path: Path,
) -> list[str]:
    issues: list[str] = []
    missing = sorted(required_cols - set(df.columns))
    if missing:
        issues.append(f"[{df_name}] Missing columns in {path}: {missing}")
    return issues


def check_panel_df(df: pd.DataFrame, path: Path) -> list[str]:
    issues: list[str] = []
    issues.extend(validate_required_columns(df, PANEL_REQUIRED_COLS, "PANEL", path))
    if issues:
        return issues

    if df.empty:
        issues.append(f"[PANEL] Empty dataframe: {path}")
        return issues

    dedup_cols = [
        "evaluation_version",
        "instrument_id",
        "symbol",
        "split_id",
        "dataset_role",
        "metric_name",
    ]
    if df.duplicated(subset=dedup_cols).any():
        issues.append(f"[PANEL] Duplicate rows found in {path}")

    expected_metric_names = {"qlike", "rmse", "mae", "macro_f1", "balanced_accuracy"}
    got_metric_names = set(df["metric_name"].dropna().unique().tolist())
    if got_metric_names != expected_metric_names:
        issues.append(
            f"[PANEL] Unexpected metric_name set in {path}. "
            f"expected={sorted(expected_metric_names)} got={sorted(got_metric_names)}"
        )

    expected_roles = {"validation", "test"}
    got_roles = set(df["dataset_role"].dropna().unique().tolist())
    if got_roles != expected_roles:
        issues.append(
            f"[PANEL] Unexpected dataset_role set in {path}. "
            f"expected={sorted(expected_roles)} got={sorted(got_roles)}"
        )

    expected_directions = {"lower_is_better", "higher_is_better"}
    got_directions = set(df["optimization_direction"].dropna().unique().tolist())
    if got_directions != expected_directions:
        issues.append(
            f"[PANEL] Unexpected optimization_direction set in {path}. "
            f"expected={sorted(expected_directions)} got={sorted(got_directions)}"
        )

    per_symbol_rows = df.groupby(["instrument_id", "symbol"]).size()
    if not (per_symbol_rows == 70).all():
        issues.append(f"[PANEL] Expected 70 panel rows per symbol in {path}")

    return issues


def check_summary_df(df: pd.DataFrame, path: Path) -> list[str]:
    issues: list[str] = []
    issues.extend(validate_required_columns(df, SUMMARY_REQUIRED_COLS, "SUMMARY", path))
    if issues:
        return issues

    if df.empty:
        issues.append(f"[SUMMARY] Empty dataframe: {path}")
        return issues

    dedup_cols = ["evaluation_version", "instrument_id", "symbol"]
    if df.duplicated(subset=dedup_cols).any():
        issues.append(f"[SUMMARY] Duplicate symbol summary rows found in {path}")

    allowed_decisions = {"promote_ml", "do_not_promote_ml"}
    got_decisions = set(df["decision"].dropna().unique().tolist())
    if not got_decisions.issubset(allowed_decisions):
        issues.append(
            f"[SUMMARY] Unexpected decision values in {path}: {sorted(got_decisions - allowed_decisions)}"
        )

    allowed_calibration = {"not_evaluable_yet", "pending_evaluation"}
    got_calibration = set(df["calibration_status"].dropna().unique().tolist())
    if not got_calibration.issubset(allowed_calibration):
        issues.append(
            f"[SUMMARY] Unexpected calibration_status values in {path}: "
            f"{sorted(got_calibration - allowed_calibration)}"
        )

    if (df["comparison_rows"].astype(int) <= 0).any():
        issues.append(f"[SUMMARY] Non-positive comparison_rows found in {path}")

    if (df["distinct_splits"].astype(int) <= 0).any():
        issues.append(f"[SUMMARY] Non-positive distinct_splits found in {path}")

    if (df["distinct_roles"].astype(int) <= 0).any():
        issues.append(f"[SUMMARY] Non-positive distinct_roles found in {path}")

    return issues


def check_reasons_df(df: pd.DataFrame, path: Path) -> list[str]:
    issues: list[str] = []
    issues.extend(validate_required_columns(df, REASONS_REQUIRED_COLS, "REASONS", path))
    if issues:
        return issues

    if df.empty:
        issues.append(f"[REASONS] Empty dataframe: {path}")
        return issues

    dedup_cols = ["evaluation_version", "instrument_id", "symbol", "rule_name"]
    if df.duplicated(subset=dedup_cols).any():
        issues.append(f"[REASONS] Duplicate reason rows found in {path}")

    expected_rule_names = {
        "qlike_relative_improvement",
        "macro_f1_no_worse",
        "balanced_accuracy_no_worse",
        "calibration_status",
    }
    got_rule_names = set(df["rule_name"].dropna().unique().tolist())
    if got_rule_names != expected_rule_names:
        issues.append(
            f"[REASONS] Unexpected rule_name set in {path}. "
            f"expected={sorted(expected_rule_names)} got={sorted(got_rule_names)}"
        )

    per_symbol_rule_count = df.groupby(["evaluation_version", "instrument_id", "symbol"]).size()
    if not (per_symbol_rule_count == 4).all():
        issues.append(f"[REASONS] Expected exactly 4 reasons per symbol in {path}")

    return issues


def check_cross_consistency(
    panel_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    reasons_df: pd.DataFrame,
) -> list[str]:
    issues: list[str] = []

    key_cols = ["evaluation_version", "instrument_id", "symbol"]

    panel_symbols = (
        panel_df[key_cols].drop_duplicates().sort_values(key_cols).reset_index(drop=True)
    )
    summary_symbols = (
        summary_df[key_cols].drop_duplicates().sort_values(key_cols).reset_index(drop=True)
    )
    reasons_symbols = (
        reasons_df[key_cols].drop_duplicates().sort_values(key_cols).reset_index(drop=True)
    )

    if not panel_symbols.equals(summary_symbols):
        issues.append("[CROSS] panel symbols != summary symbols")
    if not summary_symbols.equals(reasons_symbols):
        issues.append("[CROSS] summary symbols != reasons symbols")

    expected_comparison_rows = (
        panel_df.groupby(key_cols, as_index=False)
        .size()
        .rename(columns={"size": "expected_comparison_rows"})
    )

    merged_summary = summary_df.merge(
        expected_comparison_rows,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )
    if merged_summary["expected_comparison_rows"].isna().any():
        issues.append("[CROSS] Missing panel match for some summary rows")
    else:
        if not (
            merged_summary["comparison_rows"].astype(int)
            == merged_summary["expected_comparison_rows"].astype(int)
        ).all():
            issues.append("[CROSS] comparison_rows in summary do not match panel counts")

    reasons_pivot = (
        reasons_df[key_cols + ["rule_name", "passed"]]
        .drop_duplicates()
        .pivot(
            index=key_cols,
            columns="rule_name",
            values="passed",
        )
        .reset_index()
    )
    reasons_pivot.columns.name = None

    reasons_pivot = reasons_pivot.rename(
        columns={
            "qlike_relative_improvement": "reason_qlike_relative_improvement",
            "macro_f1_no_worse": "reason_macro_f1_no_worse",
            "balanced_accuracy_no_worse": "reason_balanced_accuracy_no_worse",
            "calibration_status": "reason_calibration_status",
        }
    )

    merged_flags = summary_df.merge(
        reasons_pivot,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    required_reason_cols = {
        "reason_qlike_relative_improvement",
        "reason_macro_f1_no_worse",
        "reason_balanced_accuracy_no_worse",
        "reason_calibration_status",
    }
    if not required_reason_cols.issubset(merged_flags.columns):
        issues.append("[CROSS] Missing expected reason columns after pivot")
        return issues

    if not (
        merged_flags["qlike_pass"].astype(bool)
        == merged_flags["reason_qlike_relative_improvement"].astype(bool)
    ).all():
        issues.append("[CROSS] qlike_pass does not match reasons table")

    if not (
        merged_flags["macro_f1_pass"].astype(bool)
        == merged_flags["reason_macro_f1_no_worse"].astype(bool)
    ).all():
        issues.append("[CROSS] macro_f1_pass does not match reasons table")

    if not (
        merged_flags["balanced_accuracy_pass"].astype(bool)
        == merged_flags["reason_balanced_accuracy_no_worse"].astype(bool)
    ).all():
        issues.append("[CROSS] balanced_accuracy_pass does not match reasons table")

    return issues


def main() -> None:
    settings = load_settings()
    decision_cfg = settings["decision"]

    decision_root = Path(decision_cfg["outputs"]["decision_dir"])

    panel_path = decision_root / "all_symbols_decision_panel_v1.parquet"
    summary_path = decision_root / "all_symbols_decision_summary_v1.parquet"
    reasons_path = decision_root / "all_symbols_decision_reasons_v1.parquet"

    issues: list[str] = []

    for path, label in [
        (panel_path, "PANEL"),
        (summary_path, "SUMMARY"),
        (reasons_path, "REASONS"),
    ]:
        if not path.exists():
            issues.append(f"[DISCOVERY] Missing {label} artifact: {path}")

    if issues:
        print("MODEL DECISION QUALITY CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)

    panel_df = pd.read_parquet(panel_path)
    summary_df = pd.read_parquet(summary_path)
    reasons_df = pd.read_parquet(reasons_path)

    issues.extend(check_panel_df(panel_df, panel_path))
    issues.extend(check_summary_df(summary_df, summary_path))
    issues.extend(check_reasons_df(reasons_df, reasons_path))
    issues.extend(check_cross_consistency(panel_df, summary_df, reasons_df))

    if issues:
        print("MODEL DECISION QUALITY CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)

    print("MODEL DECISION QUALITY CHECKS: PASS")
    print(f"[PANEL] OK -> {panel_path}")
    print(f"[SUMMARY] OK -> {summary_path}")
    print(f"[REASONS] OK -> {reasons_path}")


if __name__ == "__main__":
    main()