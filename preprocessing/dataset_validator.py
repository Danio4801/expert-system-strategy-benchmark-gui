

from dataclasses import dataclass, field
from typing import List, Literal, Tuple
import pandas as pd
import numpy as np


@dataclass
class ReadinessIssue:

    level: Literal["CRITICAL", "WARNING", "INFO"]
    code: str
    message: str
    impact: str
    recommendation: str


@dataclass
class ReadinessReport:

    score: int
    issues: List[ReadinessIssue] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    verdict: Literal["RECOMMENDED", "CAUTION", "NOT_RECOMMENDED"] = "RECOMMENDED"

    def get_critical_issues(self) -> List[ReadinessIssue]:

        return [i for i in self.issues if i.level == "CRITICAL"]

    def get_warning_issues(self) -> List[ReadinessIssue]:

        return [i for i in self.issues if i.level == "WARNING"]

    def get_info_issues(self) -> List[ReadinessIssue]:

        return [i for i in self.issues if i.level == "INFO"]

    def print_report(self) -> None:

        print("=" * 70)
        print("DATASET READINESS REPORT")
        print("=" * 70)
        print(f"\nScore: {self.score}/100", end=" ")

        if self.verdict == "RECOMMENDED":
            print("(GOOD - Dataset should work well)")
        elif self.verdict == "CAUTION":
            print("(MEDIUM - Dataset may work with limitations)")
        else:
            print("(LOW - System may not work well)")


        critical = self.get_critical_issues()
        if critical:
            print(f"\nCRITICAL Issues ({len(critical)}):")
            for issue in critical:
                print(f"  ❌ {issue.message}")
                print(f"     Impact: {issue.impact}")
                print(f"     Recommendation: {issue.recommendation}")


        warnings = self.get_warning_issues()
        if warnings:
            print(f"\nWARNING Issues ({len(warnings)}):")
            for issue in warnings:
                print(f"  ⚠️  {issue.message}")
                print(f"     Impact: {issue.impact}")
                print(f"     Recommendation: {issue.recommendation}")


        infos = self.get_info_issues()
        if infos:
            print(f"\nINFO ({len(infos)}):")
            for issue in infos:
                print(f"  ℹ️  {issue.message}")
                print(f"     Impact: {issue.impact}")


        if self.passed_checks:
            print(f"\n✅ PASSED Checks ({len(self.passed_checks)}):")
            for check in self.passed_checks:
                print(f"  - {check}")


        print(f"\nFINAL VERDICT: ", end="")
        if self.verdict == "RECOMMENDED":
            print("✅ RECOMMENDED for this system")
        elif self.verdict == "CAUTION":
            print("⚠️  USE WITH CAUTION - May have limitations")
        else:
            print("❌ NOT RECOMMENDED for this system")
            print("Consider: Converting categorical columns or using different dataset")

        print("=" * 70)


class DatasetReadinessValidator:























    def __init__(self, min_samples: int = 10, categorical_critical_threshold: float = 0.5):





        self.min_samples = min_samples
        self.categorical_critical_threshold = categorical_critical_threshold

    def validate(
        self,
        df: pd.DataFrame,
        decision_column: str
    ) -> ReadinessReport:










        score = 100
        issues = []
        passed_checks = []


        if decision_column not in df.columns:
            issues.append(ReadinessIssue(
                level="CRITICAL",
                code="DC_NOT_FOUND",
                message=f"Decision column '{decision_column}' not found in dataset",
                impact="Cannot proceed",
                recommendation=f"Check that column name is correct"
            ))
            return ReadinessReport(score=0, issues=issues, verdict="NOT_RECOMMENDED")


        feature_cols = [col for col in df.columns if col != decision_column]


        cat_score, cat_issues, cat_passed = self._check_categorical_dominance(df, feature_cols)
        score += cat_score
        issues.extend(cat_issues)
        passed_checks.extend(cat_passed)


        num_score, num_issues, num_passed = self._check_numeric_as_strings(df, feature_cols)
        score += num_score
        issues.extend(num_issues)
        passed_checks.extend(num_passed)


        size_score, size_issues, size_passed = self._check_dataset_size(df)
        score += size_score
        issues.extend(size_issues)
        passed_checks.extend(size_passed)


        miss_score, miss_issues, miss_passed = self._check_missing_values(df, decision_column)
        score += miss_score
        issues.extend(miss_issues)
        passed_checks.extend(miss_passed)


        bal_score, bal_issues, bal_passed = self._check_class_balance(df, decision_column)
        score += bal_score
        issues.extend(bal_issues)
        passed_checks.extend(bal_passed)


        const_score, const_issues, const_passed = self._check_constant_columns(df, feature_cols)
        score += const_score
        issues.extend(const_issues)
        passed_checks.extend(const_passed)


        score = max(0, min(100, score))


        if score >= 80:
            verdict = "RECOMMENDED"
        elif score >= 50:
            verdict = "CAUTION"
        else:
            verdict = "NOT_RECOMMENDED"

        return ReadinessReport(
            score=score,
            issues=issues,
            passed_checks=passed_checks,
            verdict=verdict
        )

    def _check_categorical_dominance(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[int, List[ReadinessIssue], List[str]]:






        issues = []
        passed = []
        score_delta = 0

        if not feature_cols:
            return score_delta, issues, passed


        cat_cols = df[feature_cols].select_dtypes(include=['object']).columns
        n_cat = len(cat_cols)
        n_total = len(feature_cols)
        cat_pct = (n_cat / n_total) * 100 if n_total > 0 else 0

        if cat_pct > self.categorical_critical_threshold * 100:

            issues.append(ReadinessIssue(
                level="CRITICAL",
                code="CAT_DOM",
                message=f"{n_cat} categorical columns ({cat_pct:.0f}% of dataset)",
                impact="Expected 0% accuracy - system works best with numeric data",
                recommendation="Convert categorical to numeric or use different dataset"
            ))
            score_delta = -100
        elif cat_pct > 30:

            issues.append(ReadinessIssue(
                level="WARNING",
                code="CAT_WARN",
                message=f"{n_cat} categorical columns ({cat_pct:.0f}% of dataset)",
                impact="Accuracy may be lower than expected",
                recommendation="Consider converting categorical to numeric for better results"
            ))
            score_delta = -30
        else:

            if n_cat > 0:
                passed.append(f"Categorical columns: {n_cat}/{n_total} ({cat_pct:.0f}%) - acceptable")
            else:
                passed.append(f"Dataset is fully numeric ({n_total} columns)")

        return score_delta, issues, passed

    def _check_numeric_as_strings(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[int, List[ReadinessIssue], List[str]]:










        issues = []
        passed = []
        score_delta = 0


        obj_cols = df[feature_cols].select_dtypes(include=['object']).columns

        numeric_as_string_cols = []

        for col in obj_cols:

            series = df[col].replace(['?', 'NA', 'na', '', ' ', 'nan', 'NaN'], np.nan)


            series_clean = series.dropna()

            if len(series_clean) == 0:
                continue



            numeric_converted = pd.to_numeric(series_clean, errors='coerce')
            numeric_count = numeric_converted.notna().sum()
            total_count = len(series_clean)

            numeric_ratio = numeric_count / total_count if total_count > 0 else 0

            if numeric_ratio > 0.8:

                numeric_as_string_cols.append(col)

        if numeric_as_string_cols:
            for col in numeric_as_string_cols:
                issues.append(ReadinessIssue(
                    level="WARNING",
                    code="NUM_STR",
                    message=f"Column '{col}' has numeric values stored as strings",
                    impact="Will not be discretized (treated as categorical)",
                    recommendation=f"Convert column '{col}' to numeric type (use DatasetPreparer)"
                ))
                score_delta -= 10
        else:
            if len(obj_cols) > 0:
                passed.append(f"No numeric-as-string issues detected in {len(obj_cols)} object columns")

        return score_delta, issues, passed

    def _check_dataset_size(
        self,
        df: pd.DataFrame
    ) -> Tuple[int, List[ReadinessIssue], List[str]]:






        issues = []
        passed = []
        score_delta = 0

        n_rows = len(df)

        if n_rows < self.min_samples:
            issues.append(ReadinessIssue(
                level="CRITICAL",
                code="SIZE_MIN",
                message=f"Dataset too small: {n_rows} rows (minimum {self.min_samples} required)",
                impact="Insufficient data for training",
                recommendation=f"Use dataset with at least {self.min_samples} samples"
            ))
            score_delta = -100
        elif n_rows < 100:
            issues.append(ReadinessIssue(
                level="WARNING",
                code="SIZE_SMALL",
                message=f"Small dataset: {n_rows} rows",
                impact="Results may vary, cross-validation recommended",
                recommendation="Consider using larger dataset if available"
            ))
            score_delta = -10
        else:
            passed.append(f"Dataset size: {n_rows} rows (OK)")

        return score_delta, issues, passed

    def _check_missing_values(
        self,
        df: pd.DataFrame,
        decision_column: str
    ) -> Tuple[int, List[ReadinessIssue], List[str]]:






        issues = []
        passed = []
        score_delta = 0


        decision_missing = df[decision_column].isnull().sum()
        if decision_missing > 0:
            issues.append(ReadinessIssue(
                level="WARNING",
                code="MISS_DEC",
                message=f"Decision column has {decision_missing} missing values",
                impact="Rows with missing decision will be dropped",
                recommendation="Clean decision column before processing"
            ))
            score_delta -= 10


        feature_cols = [col for col in df.columns if col != decision_column]
        total_missing = df[feature_cols].isnull().sum().sum()
        total_cells = len(df) * len(feature_cols)
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

        if missing_pct > 0:
            issues.append(ReadinessIssue(
                level="INFO",
                code="MISS_INFO",
                message=f"{total_missing} missing values ({missing_pct:.1f}% of data)",
                impact="Will use class-conditional imputation",
                recommendation="System will handle automatically"
            ))
        else:
            passed.append("No missing values in features")

        return score_delta, issues, passed

    def _check_class_balance(
        self,
        df: pd.DataFrame,
        decision_column: str
    ) -> Tuple[int, List[ReadinessIssue], List[str]]:






        issues = []
        passed = []
        score_delta = 0


        class_counts = df[decision_column].value_counts()

        if len(class_counts) < 2:
            issues.append(ReadinessIssue(
                level="CRITICAL",
                code="BAL_ONE",
                message=f"Only 1 class found: {class_counts.index[0]}",
                impact="Cannot perform classification",
                recommendation="Dataset must have at least 2 classes"
            ))
            score_delta = -100
            return score_delta, issues, passed


        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        if imbalance_ratio > 10:
            issues.append(ReadinessIssue(
                level="WARNING",
                code="BAL_IMB",
                message=f"Dataset is imbalanced: {imbalance_ratio:.1f}x ratio ({min_count} vs {max_count} samples)",
                impact="May affect prediction accuracy for minority classes",
                recommendation="Consider stratified sampling or class balancing techniques"
            ))
            score_delta -= 10
        else:
            passed.append(f"Class balance: {imbalance_ratio:.1f}x ratio (OK)")

        return score_delta, issues, passed

    def _check_constant_columns(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[int, List[ReadinessIssue], List[str]]:






        issues = []
        passed = []
        score_delta = 0

        constant_cols = []
        for col in feature_cols:
            if df[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            issues.append(ReadinessIssue(
                level="INFO",
                code="CONST_COL",
                message=f"{len(constant_cols)} constant columns: {', '.join(constant_cols[:3])}{'...' if len(constant_cols) > 3 else ''}",
                impact="Constant columns don't provide information",
                recommendation="Remove constant columns (use DatasetPreparer)"
            ))
        else:
            passed.append("No constant columns detected")

        return score_delta, issues, passed
