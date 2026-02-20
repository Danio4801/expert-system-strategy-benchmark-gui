






















import pytest
import pandas as pd
import numpy as np
from preprocessing.imputer import Imputer, ImputationReport


class TestImputerCheckMissing:

    
    def test_no_missing_values(self):

        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "Class": ["a", "a", "b"]
        })
        imputer = Imputer()
        
        result = imputer.check_missing(df)
        
        assert result == {}
    
    def test_one_column_with_missing(self):

        df = pd.DataFrame({
            "A": [1, np.nan, 3],
            "B": ["x", "y", "z"],
            "Class": ["a", "a", "b"]
        })
        imputer = Imputer()
        
        result = imputer.check_missing(df)
        
        assert result == {"A": 1}
    
    def test_multiple_columns_with_missing(self):

        df = pd.DataFrame({
            "A": [1, np.nan, 3],
            "B": ["x", None, "z"],
            "C": [np.nan, np.nan, 3],
            "Class": ["a", "a", "b"]
        })
        imputer = Imputer()
        
        result = imputer.check_missing(df)
        
        assert result == {"A": 1, "B": 1, "C": 2}


class TestImputerHasMissing:

    
    def test_no_missing_returns_false(self):

        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        imputer = Imputer()

        assert not imputer.has_missing(df)
    
    def test_with_missing_returns_true(self):

        df = pd.DataFrame({"A": [1, np.nan, 3], "B": ["x", "y", "z"]})
        imputer = Imputer()

        assert imputer.has_missing(df)
    
    def test_with_none_returns_true(self):

        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", None, "z"]})
        imputer = Imputer()

        assert imputer.has_missing(df)


class TestImputerNumericMean:


    
    def test_mean_imputation_by_class(self):

        df = pd.DataFrame({
            "Value": [10.0, np.nan, 30.0, 100.0, np.nan, 300.0],
            "Class": ["A", "A", "A", "B", "B", "B"]
        })


        
        imputer = Imputer()
        result, report = imputer.impute(df, decision_column="Class", numeric_method="mean")
        

        assert result.loc[1, "Value"] == 20.0
        assert result.loc[4, "Value"] == 200.0
        

        assert report.total_missing == 2
        assert "Value" in report.columns_affected
        assert report.values_imputed["Value"] == 2
    
    def test_mean_preserves_existing_values(self):

        df = pd.DataFrame({
            "Value": [10.0, np.nan, 30.0],
            "Class": ["A", "A", "A"]
        })
        
        imputer = Imputer()
        result, _ = imputer.impute(df, decision_column="Class", numeric_method="mean")
        
        assert result.loc[0, "Value"] == 10.0
        assert result.loc[2, "Value"] == 30.0


class TestImputerNumericMedian:


    
    def test_median_imputation_by_class(self):

        df = pd.DataFrame({
            "Value": [10.0, 20.0, np.nan, 100.0, 100.0, 200.0, 300.0, np.nan],
            "Class": ["A", "A", "A", "B", "B", "B", "B", "B"]
        })


        
        imputer = Imputer()
        result, report = imputer.impute(df, decision_column="Class", numeric_method="median")
        
        assert result.loc[2, "Value"] == 15.0
        assert result.loc[7, "Value"] == 150.0
        assert report.method_used == "median/mode"


class TestImputerCategoricalMode:


    
    def test_mode_imputation_by_class(self):

        df = pd.DataFrame({
            "Color": ["red", "red", None, "blue", "blue", "green", None],
            "Class": ["A", "A", "A", "B", "B", "B", "B"]
        })


        
        imputer = Imputer()
        result, report = imputer.impute(df, decision_column="Class", categorical_method="mode")
        
        assert result.loc[2, "Color"] == "red"
        assert result.loc[6, "Color"] == "blue"
    
    def test_mode_with_single_value_class(self):

        df = pd.DataFrame({
            "Color": ["red", None],
            "Class": ["A", "A"]
        })
        
        imputer = Imputer()
        result, _ = imputer.impute(df, decision_column="Class")
        
        assert result.loc[1, "Color"] == "red"


class TestImputerMixedTypes:


    
    def test_mixed_numeric_and_categorical(self):

        df = pd.DataFrame({
            "Age": [25.0, np.nan, 35.0, 55.0, np.nan, 65.0],
            "City": ["NYC", None, "NYC", "LA", "LA", None],
            "Class": ["Young", "Young", "Young", "Old", "Old", "Old"]
        })
        
        imputer = Imputer()
        result, report = imputer.impute(
            df, 
            decision_column="Class",
            numeric_method="mean",
            categorical_method="mode"
        )
        

        assert result.loc[1, "Age"] == 30.0
        assert result.loc[4, "Age"] == 60.0
        

        assert result.loc[1, "City"] == "NYC"
        assert result.loc[5, "City"] == "LA"
        
        assert report.total_missing == 4
        assert set(report.columns_affected) == {"Age", "City"}


class TestImputerValidation:


    
    def test_missing_decision_column_raises_error(self):

        df = pd.DataFrame({"A": [1, 2, 3]})
        imputer = Imputer()
        
        with pytest.raises(ValueError, match="nie istnieje"):
            imputer.impute(df, decision_column="NonExistent")
    
    def test_missing_values_in_decision_column_raises_error(self):

        df = pd.DataFrame({
            "A": [1, 2, 3],
            "Class": ["a", None, "b"]
        })
        imputer = Imputer()
        
        with pytest.raises(ValueError, match="brakujące wartości"):
            imputer.impute(df, decision_column="Class")


class TestImputerNoMissingValues:

    
    def test_no_changes_when_no_missing(self):

        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "Class": ["a", "a", "b"]
        })
        
        imputer = Imputer()
        result, report = imputer.impute(df, decision_column="Class")
        
        pd.testing.assert_frame_equal(result, df)
        assert report.total_missing == 0
        assert report.columns_affected == []


class TestImputerSelectiveColumns:


    
    def test_impute_only_selected_columns(self):

        df = pd.DataFrame({
            "A": [1, np.nan, 3],
            "B": [np.nan, 2, 3],
            "Class": ["x", "x", "x"]
        })
        
        imputer = Imputer()
        result, report = imputer.impute(
            df, 
            decision_column="Class",
            columns=["A"]
        )
        

        assert not pd.isna(result.loc[1, "A"])
        

        assert pd.isna(result.loc[0, "B"])
        
        assert report.columns_affected == ["A"]


class TestImputerReport:


    
    def test_report_contains_imputation_values(self):

        df = pd.DataFrame({
            "Value": [10.0, np.nan, 30.0, 100.0, np.nan],
            "Class": ["A", "A", "A", "B", "B"]
        })
        
        imputer = Imputer()
        _, report = imputer.impute(df, decision_column="Class", numeric_method="mean")
        
        assert "Value" in report.imputation_values
        assert report.imputation_values["Value"]["A"] == 20.0
        assert report.imputation_values["Value"]["B"] == 100.0
    
    def test_get_last_report(self):

        df = pd.DataFrame({
            "Value": [1, np.nan],
            "Class": ["A", "A"]
        })
        
        imputer = Imputer()
        imputer.impute(df, decision_column="Class")
        
        report = imputer.get_last_report()
        assert report is not None
        assert report.total_missing == 1


class TestImputerDropMissing:


    
    def test_drop_all_rows_with_missing(self):

        df = pd.DataFrame({
            "A": [1, np.nan, 3, 4],
            "B": ["x", "y", None, "w"],
            "C": [1, 2, 3, 4]
        })
        
        imputer = Imputer()
        result = imputer.drop_missing(df)
        
        assert len(result) == 2
        assert list(result.index) == [0, 3]
    
    def test_drop_with_threshold(self):

        df = pd.DataFrame({
            "A": [1, np.nan, np.nan],
            "B": [1, 2, np.nan],
            "C": [1, 2, np.nan],
            "D": [1, 2, np.nan]
        })



        
        imputer = Imputer()
        result = imputer.drop_missing(df, threshold=0.5)
        
        assert len(result) == 2
    
    def test_drop_from_specific_columns(self):

        df = pd.DataFrame({
            "A": [1, np.nan, 3],
            "B": [np.nan, 2, 3],
            "C": [1, 2, 3]
        })
        
        imputer = Imputer()
        result = imputer.drop_missing(df, columns=["A"])
        
        assert len(result) == 2
        assert 1 not in result.index


class TestImputerEdgeCases:


    
    def test_all_values_missing_in_column_for_one_class(self):

        df = pd.DataFrame({
            "Value": [10.0, 20.0, np.nan, np.nan],
            "Class": ["A", "A", "B", "B"]
        })

        
        imputer = Imputer()
        result, _ = imputer.impute(df, decision_column="Class", numeric_method="mean")
        


        assert not pd.isna(result.loc[0, "Value"])
        assert not pd.isna(result.loc[1, "Value"])

    
    def test_single_row_per_class(self):

        df = pd.DataFrame({
            "Value": [10.0, np.nan],
            "Class": ["A", "B"]
        })
        
        imputer = Imputer()
        result, report = imputer.impute(df, decision_column="Class", numeric_method="mean")
        


        assert report.total_missing == 1
    
    def test_empty_dataframe(self):

        df = pd.DataFrame(columns=["A", "Class"])
        
        imputer = Imputer()
        result, report = imputer.impute(df, decision_column="Class")
        
        assert len(result) == 0
        assert report.total_missing == 0
