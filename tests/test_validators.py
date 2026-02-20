















import pytest
import pandas as pd
import numpy as np
from preprocessing.validators import (
    validate_decision_column,
    ValidationResult,
    ValidationError
)


class TestValidateDecisionColumn:



    def test_dc01_column_not_found(self):

        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"]
        })

        _, validation = validate_decision_column(df, decision_column="NonExistent")

        assert not validation.is_valid
        assert len(validation.errors) == 1
        assert validation.errors[0].code == "DC01"
        assert "nie istnieje" in validation.errors[0].message
        assert validation.errors[0].is_critical

    def test_dc01_index_out_of_range(self):

        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"]
        })


        _, validation = validate_decision_column(df, decision_column=5)

        assert not validation.is_valid
        assert len(validation.errors) == 1
        assert validation.errors[0].code == "DC01"
        assert "poza zakresem" in validation.errors[0].message

    def test_dc01_index_valid(self):

        df = pd.DataFrame({
            "A": [1, 2, 3],
            "Class": ["a", "b", "c"]
        })


        df_clean, validation = validate_decision_column(df, decision_column=1, min_rows=3)

        assert validation.is_valid
        assert len(validation.errors) == 0

    def test_dc02_nulls_with_drop_true(self):

        df = pd.DataFrame({
            "A": [1, 2, 3, 4],
            "Class": ["a", None, "b", "c"]
        })

        df_clean, validation = validate_decision_column(
            df,
            decision_column="Class",
            drop_missing=True,
            min_rows=3
        )


        assert len(df_clean) == 3
        assert not df_clean["Class"].isnull().any()


        assert len(validation.infos) == 1
        assert "Usunięto 1 wierszy" in validation.infos[0]
        assert "DF02" in validation.infos[0]


        assert validation.is_valid

    def test_dc02_nulls_with_drop_false(self):

        df = pd.DataFrame({
            "A": [1, 2, 3],
            "Class": ["a", None, "b"]
        })

        _, validation = validate_decision_column(
            df,
            decision_column="Class",
            drop_missing=False,
            min_rows=3
        )

        assert not validation.is_valid
        assert len(validation.errors) == 1
        assert validation.errors[0].code == "DC02"
        assert "pustych wartości" in validation.errors[0].message

    def test_dc03_single_class(self):

        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "Class": ["a", "a", "a", "a", "a"]
        })

        _, validation = validate_decision_column(df, decision_column="Class", min_rows=5)

        assert not validation.is_valid
        assert len(validation.errors) == 1
        assert validation.errors[0].code == "DC03"
        assert "tylko 1 unikalną klasę" in validation.errors[0].message

    def test_dc04_imbalanced_warning(self):

        df = pd.DataFrame({
            "A": list(range(100)),

            "Class": ["a"] * 95 + ["b"] * 5
        })

        _, validation = validate_decision_column(
            df,
            decision_column="Class",
            imbalance_threshold=0.9
        )


        assert validation.is_valid
        assert len(validation.warnings) == 1
        assert "DC04" in validation.warnings[0]
        assert "Niezbalansowane klasy" in validation.warnings[0]
        assert "95.0%" in validation.warnings[0]

    def test_dc04_balanced_no_warning(self):

        df = pd.DataFrame({
            "A": list(range(100)),

            "Class": ["a"] * 50 + ["b"] * 50
        })

        _, validation = validate_decision_column(
            df,
            decision_column="Class",
            imbalance_threshold=0.9
        )

        assert validation.is_valid
        assert len(validation.warnings) == 0

    def test_df01_too_few_rows(self):

        df = pd.DataFrame({
            "A": [1, 2, 3],
            "Class": ["a", "b", "c"]
        })

        _, validation = validate_decision_column(
            df,
            decision_column="Class",
            min_rows=10
        )

        assert not validation.is_valid
        assert len(validation.errors) == 1
        assert validation.errors[0].code == "DF01"
        assert "Za mało wierszy" in validation.errors[0].message
        assert "3 < 10" in validation.errors[0].message

    def test_df02_info_rows_dropped(self):

        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "Class": ["a", None, "b", None, "c"]
        })

        df_clean, validation = validate_decision_column(
            df,
            decision_column="Class",
            drop_missing=True,
            min_rows=3
        )


        assert len(df_clean) == 3


        assert len(validation.infos) == 1
        assert "DF02" in validation.infos[0]
        assert "Usunięto 2 wierszy" in validation.infos[0]

    def test_multiple_errors(self):

        df = pd.DataFrame({
            "A": [1, 2],
            "Class": ["a", "a"]
        })

        _, validation = validate_decision_column(
            df,
            decision_column="Class",
            min_rows=10
        )

        assert not validation.is_valid

        assert len(validation.errors) == 2
        error_codes = {err.code for err in validation.errors}
        assert "DC03" in error_codes
        assert "DF01" in error_codes

    def test_valid_dataframe(self):

        df = pd.DataFrame({
            "A": list(range(100)),
            "Class": ["a"] * 50 + ["b"] * 50
        })

        df_clean, validation = validate_decision_column(
            df,
            decision_column="Class"
        )

        assert validation.is_valid
        assert len(validation.errors) == 0
        assert len(validation.warnings) == 0
        assert len(df_clean) == 100

    def test_custom_min_rows(self):

        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "Class": ["a", "b", "c", "a", "b"]
        })


        _, validation = validate_decision_column(
            df,
            decision_column="Class",
            min_rows=5
        )
        assert validation.is_valid


        _, validation = validate_decision_column(
            df,
            decision_column="Class",
            min_rows=6
        )
        assert not validation.is_valid
        assert validation.errors[0].code == "DF01"

    def test_custom_imbalance_threshold(self):

        df = pd.DataFrame({
            "A": list(range(100)),

            "Class": ["a"] * 80 + ["b"] * 20
        })


        _, validation = validate_decision_column(
            df,
            decision_column="Class",
            imbalance_threshold=0.9
        )
        assert validation.is_valid
        assert len(validation.warnings) == 0


        _, validation = validate_decision_column(
            df,
            decision_column="Class",
            imbalance_threshold=0.7
        )
        assert validation.is_valid
        assert len(validation.warnings) == 1
        assert "DC04" in validation.warnings[0]

    def test_empty_dataframe(self):

        df = pd.DataFrame(columns=["A", "Class"])

        _, validation = validate_decision_column(
            df,
            decision_column="Class",
            min_rows=1
        )

        assert not validation.is_valid

        error_codes = {err.code for err in validation.errors}
        assert "DF01" in error_codes
        assert "DC03" in error_codes
