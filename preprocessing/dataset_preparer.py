

from dataclasses import dataclass, field
from typing import List, Set
import pandas as pd
import numpy as np


@dataclass
class PreparedDataset:

    df: pd.DataFrame
    changes_log: List[str] = field(default_factory=list)

    def print_summary(self) -> None:

        print("=" * 70)
        print("DATASET PREPARATION SUMMARY")
        print("=" * 70)

        if not self.changes_log:
            print("\n✅ No changes needed - dataset is ready to use")
        else:
            print(f"\n📝 {len(self.changes_log)} changes applied:\n")
            for i, change in enumerate(self.changes_log, 1):

                if "Converted" in change and "numeric" in change:
                    icon = "🔢"
                elif "Removed" in change:
                    icon = "🗑️"
                elif "Replaced" in change:
                    icon = "🔄"
                elif "Stripped" in change:
                    icon = "✂️"
                else:
                    icon = "✅"

                print(f"  {i}. {icon} {change}")

        print(f"\n📊 Final dataset shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print("=" * 70)


class DatasetPreparer:
















    DEFAULT_MISSING_MARKERS = {'?', 'NA', 'na', 'N/A', 'n/a', '', ' ', 'nan', 'NaN', 'NULL', 'null'}

    def __init__(
        self,
        missing_markers: Set[str] = None,
        numeric_threshold: float = 0.8,
        remove_constant: bool = True,
        strip_whitespace: bool = True
    ):







        self.missing_markers = missing_markers if missing_markers is not None else self.DEFAULT_MISSING_MARKERS
        self.numeric_threshold = numeric_threshold
        self.remove_constant = remove_constant
        self.strip_whitespace = strip_whitespace

    def prepare(
        self,
        df: pd.DataFrame,
        decision_column: str
    ) -> PreparedDataset:











        df = df.copy()
        changes_log = []


        if decision_column not in df.columns:
            raise ValueError(f"Decision column '{decision_column}' not found in dataset")

        feature_cols = [col for col in df.columns if col != decision_column]


        marker_changes = self._replace_missing_markers(df, feature_cols)
        changes_log.extend(marker_changes)


        if self.strip_whitespace:
            strip_changes = self._strip_whitespace(df, df.columns)
            changes_log.extend(strip_changes)


        numeric_changes = self._convert_numeric_as_string(df, feature_cols)
        changes_log.extend(numeric_changes)


        if self.remove_constant:
            constant_changes, df = self._remove_constant_columns(df, feature_cols)
            changes_log.extend(constant_changes)

        return PreparedDataset(df=df, changes_log=changes_log)

    def _replace_missing_markers(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> List[str]:






        changes = []

        for col in feature_cols:

            if df[col].dtype != 'object':
                continue


            mask = df[col].isin(self.missing_markers)
            n_replaced = mask.sum()

            if n_replaced > 0:
                df[col] = df[col].replace(list(self.missing_markers), np.nan)
                changes.append(
                    f"Replaced {n_replaced} missing markers in '{col}' with NaN"
                )

        return changes

    def _strip_whitespace(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> List[str]:






        changes = []

        for col in columns:

            if df[col].dtype != 'object':
                continue


            original_values = df[col].copy()
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)


            changed = (original_values != df[col]).sum()
            if changed > 0:
                changes.append(
                    f"Stripped whitespace from {changed} values in '{col}'"
                )

        return changes

    def _convert_numeric_as_string(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> List[str]:










        changes = []

        for col in feature_cols:

            if df[col].dtype != 'object':
                continue


            series_clean = df[col].dropna()

            if len(series_clean) == 0:
                continue


            numeric_converted = pd.to_numeric(series_clean, errors='coerce')
            numeric_count = numeric_converted.notna().sum()
            total_count = len(series_clean)

            numeric_ratio = numeric_count / total_count if total_count > 0 else 0

            if numeric_ratio >= self.numeric_threshold:


                df[col] = pd.to_numeric(df[col], errors='coerce')

                changes.append(
                    f"Converted '{col}' to numeric ({numeric_count}/{total_count} values = {numeric_ratio*100:.1f}%)"
                )

        return changes

    def _remove_constant_columns(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> tuple[List[str], pd.DataFrame]:






        changes = []
        cols_to_drop = []

        for col in feature_cols:
            if df[col].nunique() <= 1:
                cols_to_drop.append(col)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            changes.append(
                f"Removed {len(cols_to_drop)} constant columns: {', '.join(cols_to_drop)}"
            )

        return changes, df

    def prepare_with_validation(
        self,
        df: pd.DataFrame,
        decision_column: str
    ) -> tuple[PreparedDataset, 'ReadinessReport']:






        from preprocessing.dataset_validator import DatasetReadinessValidator


        prepared = self.prepare(df, decision_column)


        validator = DatasetReadinessValidator()
        report = validator.validate(prepared.df, decision_column)

        return prepared, report
