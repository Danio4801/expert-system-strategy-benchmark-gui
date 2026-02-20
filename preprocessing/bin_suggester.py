










from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np
import logging


default_logger = logging.getLogger(__name__)


@dataclass
class ColumnStats:

    n: int
    min_val: float
    max_val: float
    std: float
    iqr: float
    skewness: float
    has_outliers: bool
    unique_ratio: float


@dataclass
class BinSuggestion:

    sturges: int
    scott: int
    freedman_diaconis: int
    recommended: str
    recommended_bins: int
    reasons: List[str]


class BinSuggester:














    def __init__(self, min_bins: int = 3, max_bins: int = 20, logger: Optional[logging.Logger] = None):







        self.min_bins = min_bins
        self.max_bins = max_bins
        self.logger = logger if logger else default_logger

    def analyze_column(self, series: pd.Series) -> ColumnStats:









        clean_series = series.dropna()

        if len(clean_series) == 0:
            raise ValueError("Kolumna nie zawiera wartości (same NaN)")


        n = len(clean_series)
        min_val = float(clean_series.min())
        max_val = float(clean_series.max())
        std = float(clean_series.std())


        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = float(q3 - q1)


        skewness = float(clean_series.skew())


        has_outliers = self._detect_outliers_iqr(clean_series)


        unique_count = clean_series.nunique()
        unique_ratio = unique_count / n

        return ColumnStats(
            n=n,
            min_val=min_val,
            max_val=max_val,
            std=std,
            iqr=iqr,
            skewness=skewness,
            has_outliers=has_outliers,
            unique_ratio=unique_ratio
        )

    def suggest(self, series: pd.Series) -> BinSuggestion:












        stats = self.analyze_column(series)


        self.logger.debug(f"[BINNING] Analyzing column: n={stats.n}, skewness={stats.skewness:.3f}, has_outliers={stats.has_outliers}")
        if stats.has_outliers:

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (series < lower_bound) | (series > upper_bound)
            outlier_count = outliers.sum()
            self.logger.debug(f"[BINNING] Outliers detected: {outlier_count} values (IQR method)")


        sturges = self._calculate_sturges(stats.n)
        scott = self._calculate_scott(series.dropna(), stats)
        fd = self._calculate_freedman_diaconis(series.dropna(), stats)


        self.logger.debug(f"[BINNING] Calculated bins: Sturges={sturges}, Scott={scott}, FD={fd}")


        recommended_method, reasons = self._apply_rules(stats)


        self.logger.debug(f"[BINNING] Decision: {recommended_method} method selected. Reasons: {', '.join(reasons)}")


        if recommended_method == "sturges":
            recommended_bins = sturges
        elif recommended_method == "scott":
            recommended_bins = scott
        else:
            recommended_bins = fd


        original_recommended_bins = recommended_bins
        if recommended_bins < self.min_bins:
            reasons.append(f"zwiększono do minimum {self.min_bins} binów (R12)")
            recommended_bins = self.min_bins
            self.logger.debug(f"[BINNING] R12 applied: {original_recommended_bins} -> {recommended_bins} (min constraint)")
        elif recommended_bins > self.max_bins:
            reasons.append(f"zmniejszono do maksimum {self.max_bins} binów (R13)")
            recommended_bins = self.max_bins
            self.logger.debug(f"[BINNING] R13 applied: {original_recommended_bins} -> {recommended_bins} (max constraint)")


        if stats.unique_ratio < 0.05:
            unique_count = int(stats.n * stats.unique_ratio)
            if recommended_bins > unique_count:
                original_bins = recommended_bins
                recommended_bins = max(self.min_bins, unique_count)
                reasons.append(f"zmniejszono do {unique_count} (R10: mało unikalnych wartości)")
                self.logger.debug(f"[BINNING] R10 applied: {original_bins} -> {recommended_bins} (unique_ratio={stats.unique_ratio:.3f})")


        self.logger.info(f"[BINNING] Final recommendation: {recommended_bins} bins using {recommended_method} method")

        return BinSuggestion(
            sturges=sturges,
            scott=scott,
            freedman_diaconis=fd,
            recommended=recommended_method,
            recommended_bins=recommended_bins,
            reasons=reasons
        )

    def _calculate_sturges(self, n: int) -> int:










        return int(np.ceil(1 + np.log2(n)))

    def _calculate_scott(self, series: pd.Series, stats: ColumnStats) -> int:











        if stats.std == 0:
            return self.min_bins

        bin_width = 3.5 * stats.std / (stats.n ** (1/3))
        if bin_width == 0:
            return self.min_bins

        bins = (stats.max_val - stats.min_val) / bin_width
        return max(self.min_bins, int(np.ceil(bins)))

    def _calculate_freedman_diaconis(self, series: pd.Series, stats: ColumnStats) -> int:











        if stats.iqr == 0:
            return self.min_bins

        bin_width = 2 * stats.iqr / (stats.n ** (1/3))
        if bin_width == 0:
            return self.min_bins

        bins = (stats.max_val - stats.min_val) / bin_width
        return max(self.min_bins, int(np.ceil(bins)))

    def _detect_outliers_iqr(self, series: pd.Series) -> bool:










        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers.any()

    def _apply_rules(self, stats: ColumnStats) -> tuple[str, List[str]]:













        reasons = []


        if stats.has_outliers:
            reasons.append("wykryto wartości odstające (metoda IQR) - R4")

            if stats.n < 30:
                reasons.append(f"mały dataset z outliers (n={stats.n}) - R5")


            return ("freedman_diaconis", reasons)


        if abs(stats.skewness) >= 0.5:
            reasons.append(f"rozkład skośny (skewness={stats.skewness:.2f}) - R9")
            return ("freedman_diaconis", reasons)


        if abs(stats.skewness) < 0.5:
            reasons.append(f"rozkład symetryczny (skewness={stats.skewness:.2f}) - R6/R7")


            if stats.n >= 30:
                reasons.append(f"średni/duży dataset (n={stats.n}) - R8")
                return ("scott", reasons)


        if stats.n < 30:
            reasons.append(f"mały dataset (n={stats.n}) - R1")
            return ("sturges", reasons)

        if 30 <= stats.n < 200:
            reasons.append(f"średni dataset (n={stats.n}) - R2")
            return ("scott", reasons)


        reasons.append(f"duży dataset (n={stats.n}) - R3")
        return ("freedman_diaconis", reasons)

    def suggest_for_dataframe(self, df: pd.DataFrame,
                             exclude_columns: List[str] = None) -> dict:









        exclude_columns = exclude_columns or []
        suggestions = {}

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col not in exclude_columns:
                try:
                    suggestion = self.suggest(df[col])
                    suggestions[col] = suggestion
                except Exception as e:

                    continue

        return suggestions
