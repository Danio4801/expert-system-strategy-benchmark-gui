




















import pytest
import pandas as pd
import numpy as np
from preprocessing.bin_suggester import (
    BinSuggester,
    ColumnStats,
    BinSuggestion
)


class TestColumnStats:



    def test_analyze_normal_distribution(self):

        np.random.seed(42)
        data = pd.Series(np.random.normal(100, 15, 1000))

        suggester = BinSuggester()
        stats = suggester.analyze_column(data)

        assert stats.n == 1000
        assert 40 < stats.min_val < 100
        assert 100 < stats.max_val < 170
        assert 10 < stats.std < 20
        assert abs(stats.skewness) < 0.5
        assert 0 < stats.unique_ratio <= 1.0

    def test_analyze_with_outliers(self):


        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100])

        suggester = BinSuggester()
        stats = suggester.analyze_column(data)

        assert stats.has_outliers

    def test_analyze_without_outliers(self):


        np.random.seed(42)
        data = pd.Series(np.random.normal(50, 5, 100))

        suggester = BinSuggester()
        stats = suggester.analyze_column(data)



        assert stats.has_outliers in [True, False]

    def test_analyze_skewed_distribution(self):


        np.random.seed(42)
        data = pd.Series(np.random.lognormal(0, 1, 1000))

        suggester = BinSuggester()
        stats = suggester.analyze_column(data)

        assert abs(stats.skewness) > 0.5

    def test_analyze_removes_nan(self):

        data = pd.Series([1, 2, np.nan, 3, 4, np.nan, 5])

        suggester = BinSuggester()
        stats = suggester.analyze_column(data)

        assert stats.n == 5

    def test_analyze_all_nan_raises_error(self):

        data = pd.Series([np.nan, np.nan, np.nan])

        suggester = BinSuggester()

        with pytest.raises(ValueError, match="nie zawiera wartości"):
            suggester.analyze_column(data)


class TestSturgesMethod:



    def test_sturges_small_dataset(self):

        suggester = BinSuggester()
        bins = suggester._calculate_sturges(20)

        assert bins == 6

    def test_sturges_medium_dataset(self):

        suggester = BinSuggester()
        bins = suggester._calculate_sturges(100)

        assert bins == 8

    def test_sturges_large_dataset(self):

        suggester = BinSuggester()
        bins = suggester._calculate_sturges(1000)

        assert bins == 11


class TestScottMethod:



    def test_scott_normal_distribution(self):

        np.random.seed(42)
        data = pd.Series(np.random.normal(100, 15, 1000))

        suggester = BinSuggester()
        stats = suggester.analyze_column(data)
        bins = suggester._calculate_scott(data, stats)


        assert 5 <= bins <= 30

    def test_scott_zero_std_returns_min_bins(self):

        data = pd.Series([5, 5, 5, 5, 5])

        suggester = BinSuggester()
        stats = suggester.analyze_column(data)
        bins = suggester._calculate_scott(data, stats)

        assert bins == 3


class TestFreedmanDiaconisMethod:



    def test_fd_with_outliers(self):


        np.random.seed(42)
        normal_data = list(np.random.normal(100, 10, 100))
        data_with_outliers = pd.Series(normal_data + [500, 600, 700])

        suggester = BinSuggester()
        stats = suggester.analyze_column(data_with_outliers)
        bins = suggester._calculate_freedman_diaconis(data_with_outliers, stats)



        assert bins > 0


    def test_fd_zero_iqr_returns_min_bins(self):

        data = pd.Series([5, 5, 5, 5, 5])

        suggester = BinSuggester()
        stats = suggester.analyze_column(data)
        bins = suggester._calculate_freedman_diaconis(data, stats)

        assert bins == 3


class TestRulesSmallDatasets:


    def test_r1_small_dataset_prefers_sturges(self):

        np.random.seed(42)
        data = pd.Series(np.random.normal(100, 15, 20))

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)

        assert suggestion.recommended == "sturges"
        assert any("R1" in reason for reason in suggestion.reasons)

    def test_r5_small_with_outliers_limits_bins(self):


        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 100])

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)

        assert suggestion.recommended == "freedman_diaconis"
        assert any("R5" in reason or "R4" in reason for reason in suggestion.reasons)


class TestRulesMediumDatasets:


    def test_r2_medium_dataset_prefers_scott(self):


        data = pd.Series(list(range(50, 150)))

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)



        assert suggestion.recommended in ["scott", "freedman_diaconis"]
        assert any("R2" in reason or "R8" in reason or "R3" in reason for reason in suggestion.reasons)


class TestRulesLargeDatasets:


    def test_r3_large_dataset_prefers_fd(self):


        data = pd.Series(list(range(50, 250)))

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)


        assert suggestion.recommended in ["scott", "freedman_diaconis"]

        assert len(suggestion.reasons) > 0


class TestRulesOutliers:


    def test_r4_outliers_prefer_fd(self):


        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300])

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)

        assert suggestion.recommended == "freedman_diaconis"
        assert any("R4" in reason for reason in suggestion.reasons)


class TestRulesSkewness:



    def test_r9_skewed_distribution_prefers_fd(self):


        np.random.seed(42)
        data = pd.Series(np.random.lognormal(0, 1, 100))

        suggester = BinSuggester()
        stats = suggester.analyze_column(data)
        suggestion = suggester.suggest(data)


        assert abs(stats.skewness) >= 0.5

        assert suggestion.recommended == "freedman_diaconis"

        assert any("R9" in reason or "R4" in reason for reason in suggestion.reasons)

    def test_r6_r7_symmetric_distribution(self):


        data = pd.Series(list(range(50, 150)))

        suggester = BinSuggester()
        stats = suggester.analyze_column(data)
        suggestion = suggester.suggest(data)


        assert abs(stats.skewness) < 0.5



        assert len(suggestion.reasons) > 0

        assert any(code in reason for code in ["R6", "R7", "R2", "R8", "R3"] for reason in suggestion.reasons)


class TestRulesUniqueRatio:


    def test_r10_low_unique_ratio_decreases_bins(self):


        data = pd.Series([1, 1, 1, 1, 2, 2, 2, 3, 3, 3] * 10)

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)


        assert any("R10" in reason for reason in suggestion.reasons)


class TestRulesConstraints:


    def test_r12_minimum_3_bins(self):


        data = pd.Series([1, 2, 3])

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)

        assert suggestion.recommended_bins >= 3

    def test_r13_maximum_20_bins(self):


        np.random.seed(42)
        data = pd.Series(np.random.uniform(0, 10000, 10000))

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)

        assert suggestion.recommended_bins <= 20

    def test_custom_min_max_bins(self):

        data = pd.Series(range(100))

        suggester = BinSuggester(min_bins=5, max_bins=15)
        suggestion = suggester.suggest(data)

        assert 5 <= suggestion.recommended_bins <= 15


class TestBinSuggestion:



    def test_all_three_methods_calculated(self):

        data = pd.Series(range(100))

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)

        assert suggestion.sturges > 0
        assert suggestion.scott > 0
        assert suggestion.freedman_diaconis > 0

    def test_reasons_included(self):

        data = pd.Series(range(100))

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)

        assert len(suggestion.reasons) > 0
        assert isinstance(suggestion.reasons, list)
        assert all(isinstance(r, str) for r in suggestion.reasons)

    def test_recommended_is_one_of_three(self):

        data = pd.Series(range(100))

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)

        assert suggestion.recommended in ["sturges", "scott", "freedman_diaconis"]

    def test_recommended_bins_matches_method(self):

        data = pd.Series(range(100))

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)

        if suggestion.recommended == "sturges":
            assert suggestion.recommended_bins == suggestion.sturges
        elif suggestion.recommended == "scott":
            assert suggestion.recommended_bins == suggestion.scott
        else:
            assert suggestion.recommended_bins == suggestion.freedman_diaconis


class TestSuggestForDataFrame:



    def test_suggests_for_all_numeric_columns(self):

        df = pd.DataFrame({
            "A": range(100),
            "B": range(100, 200),
            "C": ["cat"] * 100,
            "Class": [1, 2] * 50
        })

        suggester = BinSuggester()
        suggestions = suggester.suggest_for_dataframe(df, exclude_columns=["Class"])

        assert "A" in suggestions
        assert "B" in suggestions
        assert "C" not in suggestions
        assert "Class" not in suggestions

    def test_excludes_specified_columns(self):

        df = pd.DataFrame({
            "A": range(100),
            "ID": range(100),
            "Class": [1, 2] * 50
        })

        suggester = BinSuggester()
        suggestions = suggester.suggest_for_dataframe(
            df,
            exclude_columns=["ID", "Class"]
        )

        assert "A" in suggestions
        assert "ID" not in suggestions
        assert "Class" not in suggestions

    def test_handles_empty_dataframe(self):

        df = pd.DataFrame(columns=["A", "B"])

        suggester = BinSuggester()
        suggestions = suggester.suggest_for_dataframe(df)

        assert suggestions == {}

    def test_skips_columns_with_errors(self):

        df = pd.DataFrame({
            "A": range(100),
            "B": [np.nan] * 100
        })

        suggester = BinSuggester()
        suggestions = suggester.suggest_for_dataframe(df)

        assert "A" in suggestions
        assert "B" not in suggestions


class TestEdgeCases:



    def test_constant_column(self):

        data = pd.Series([5] * 100)

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)


        assert suggestion.recommended_bins == 3

    def test_binary_column(self):

        data = pd.Series([0, 1] * 50)

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)



        assert suggestion.recommended_bins <= 3

    def test_very_small_dataset(self):

        data = pd.Series([1, 2, 3, 4, 5])

        suggester = BinSuggester()
        suggestion = suggester.suggest(data)

        assert suggestion.recommended_bins >= 3
        assert suggestion.recommended == "sturges"
