




















import pytest
import pandas as pd
import numpy as np

from preprocessing.discretizer import Discretizer


class TestDiscretizerEqualWidth:

    
    def test_equal_width_creates_correct_number_of_bins(self):

        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=5)
        
        unique_values = result["value"].nunique()
        assert unique_values <= 5
    
    def test_equal_width_handles_single_column(self):

        df = pd.DataFrame({
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
            "category": ["a", "b", "c", "d", "e"]
        })
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=3, columns=["numeric"])
        

        assert result["numeric"].dtype == object or pd.api.types.is_categorical_dtype(result["numeric"])

        assert list(result["category"]) == ["a", "b", "c", "d", "e"]
    
    def test_equal_width_bins_have_equal_range(self):

        df = pd.DataFrame({"value": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=5)
        


        assert result["value"].iloc[0] != result["value"].iloc[-1]
    
    def test_equal_width_with_two_bins(self):

        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=2)
        
        assert result["value"].nunique() == 2


class TestDiscretizerEqualFrequency:

    
    def test_equal_frequency_creates_balanced_bins(self):

        df = pd.DataFrame({"value": list(range(100))})
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_frequency", bins=4)
        

        value_counts = result["value"].value_counts()
        assert all(20 <= count <= 30 for count in value_counts)
    
    def test_equal_frequency_handles_duplicates(self):


        df = pd.DataFrame({"value": [1, 1, 1, 1, 5, 5, 5, 5, 10, 10]})
        
        discretizer = Discretizer()

        result = discretizer.discretize(df, method="equal_frequency", bins=3)
        
        assert len(result) == 10
    
    def test_equal_frequency_with_two_bins(self):

        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_frequency", bins=2)
        

        assert result["value"].nunique() == 2


class TestDiscretizerKMeans:

    
    def test_kmeans_creates_correct_number_of_clusters(self):

        df = pd.DataFrame({"value": [1, 2, 3, 10, 11, 12, 20, 21, 22]})
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="kmeans", bins=3)
        
        assert result["value"].nunique() == 3
    
    def test_kmeans_groups_similar_values(self):


        df = pd.DataFrame({"value": [1, 2, 3, 100, 101, 102, 200, 201, 202]})
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="kmeans", bins=3)
        

        assert result["value"].iloc[0] == result["value"].iloc[1] == result["value"].iloc[2]

        assert result["value"].iloc[3] == result["value"].iloc[4] == result["value"].iloc[5]


class TestDiscretizerGeneral:


    
    def test_only_numeric_columns_discretized(self):

        df = pd.DataFrame({
            "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "numeric2": [10, 20, 30, 40, 50],
            "text": ["a", "b", "c", "d", "e"],
            "category": pd.Categorical(["x", "y", "z", "x", "y"])
        })
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=3)
        

        assert list(result["text"]) == ["a", "b", "c", "d", "e"]
    
    def test_preserves_row_count(self):

        df = pd.DataFrame({"value": range(100)})
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=5)
        
        assert len(result) == len(df)
    
    def test_preserves_column_count(self):

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["x", "y", "z"]
        })
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=2)
        
        assert len(result.columns) == len(df.columns)
    
    def test_specific_columns_only(self):

        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=2, columns=["a"])
        

        assert result["a"].nunique() <= 2

        assert result["b"].nunique() == 5
    
    def test_invalid_method_raises_error(self):

        df = pd.DataFrame({"value": [1, 2, 3]})
        
        discretizer = Discretizer()
        
        with pytest.raises(ValueError):
            discretizer.discretize(df, method="invalid_method", bins=2)
    
    def test_bins_must_be_positive(self):

        df = pd.DataFrame({"value": [1, 2, 3]})
        
        discretizer = Discretizer()
        
        with pytest.raises(ValueError):
            discretizer.discretize(df, method="equal_width", bins=0)
    
    def test_empty_dataframe(self):

        df = pd.DataFrame({"value": []})
        
        discretizer = Discretizer()
        
        with pytest.raises(ValueError):
            discretizer.discretize(df, method="equal_width", bins=3)


class TestDiscretizerLabels:


    
    def test_labels_are_readable(self):

        df = pd.DataFrame({"value": [1, 5, 10, 15, 20]})
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=3)
        

        assert result["value"].dtype == object or pd.api.types.is_categorical_dtype(result["value"])
    
    def test_labels_indicate_bin_number_or_range(self):

        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        
        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=2)
        

        labels = result["value"].unique()
        assert len(labels) == 2

class TestDiscretizerSkipBinary:



    def test_skip_binary_by_default(self):

        df = pd.DataFrame({
            "binary": [0, 1, 0, 1, 0],
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=3)


        assert list(result["binary"]) == [0, 1, 0, 1, 0]

        assert result["numeric"].dtype == object

    def test_skip_binary_false_discretizes_binary(self):

        df = pd.DataFrame({
            "binary": [0, 1, 0, 1, 0],
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=3, skip_binary=False)


        assert result["binary"].dtype == object

    def test_skip_binary_with_yes_no_values(self):

        df = pd.DataFrame({
            "answer": ["yes", "no", "yes", "no", "yes"],
            "score": [10, 20, 30, 40, 50]
        })

        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=3)


        assert list(result["answer"]) == ["yes", "no", "yes", "no", "yes"]


class TestDiscretizerFitTransform:



    def test_fit_stores_bin_edges(self):

        train_df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        discretizer = Discretizer()
        discretizer.fit(train_df, method="equal_width", bins=3)


        assert hasattr(discretizer, '_bin_edges')
        assert 'value' in discretizer._bin_edges

    def test_transform_uses_same_bins_as_fit(self):

        train_df = pd.DataFrame({"value": [0, 10, 20, 30, 40, 50]})
        test_df = pd.DataFrame({"value": [5, 15, 25, 35, 45]})

        discretizer = Discretizer()
        discretizer.fit(train_df, method="equal_width", bins=3)
        train_result = discretizer.transform(train_df)
        test_result = discretizer.transform(test_df)



        assert test_result["value"].iloc[0] == train_result["value"].iloc[0]

    def test_fit_transform_equivalent_to_fit_then_transform(self):

        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        discretizer1 = Discretizer()
        result1 = discretizer1.fit_transform(df, method="equal_width", bins=3)

        discretizer2 = Discretizer()
        discretizer2.fit(df, method="equal_width", bins=3)
        result2 = discretizer2.transform(df)

        assert list(result1["value"]) == list(result2["value"])

    def test_transform_before_fit_raises_error(self):

        df = pd.DataFrame({"value": [1, 2, 3]})

        discretizer = Discretizer()

        with pytest.raises(ValueError, match="nie został dopasowany"):
            discretizer.transform(df)

    def test_fit_equal_width_consistent_bins(self):


        train_df = pd.DataFrame({"value": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        test_df = pd.DataFrame({"value": [25, 50, 75]})

        discretizer = Discretizer()
        discretizer.fit(train_df, method="equal_width", bins=2)
        result = discretizer.transform(test_df)


        assert result["value"].iloc[0] == "bin_1"

        assert result["value"].iloc[2] == "bin_2"

    def test_fit_equal_frequency_consistent_bins(self):

        train_df = pd.DataFrame({"value": list(range(100))})
        test_df = pd.DataFrame({"value": [10, 30, 50, 70, 90]})

        discretizer = Discretizer()
        discretizer.fit(train_df, method="equal_frequency", bins=4)
        result = discretizer.transform(test_df)


        assert all(val.startswith("bin_") for val in result["value"])

    def test_fit_kmeans_consistent_bins(self):

        train_df = pd.DataFrame({"value": [1, 2, 3, 100, 101, 102]})
        test_df = pd.DataFrame({"value": [2.5, 101.5]})

        discretizer = Discretizer()
        discretizer.fit(train_df, method="kmeans", bins=2)
        result = discretizer.transform(test_df)


        assert all(val.startswith("cluster_") for val in result["value"])

    def test_fit_multiple_columns(self):

        train_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })

        discretizer = Discretizer()
        discretizer.fit(train_df, method="equal_width", bins=2)


        assert 'a' in discretizer._bin_edges
        assert 'b' in discretizer._bin_edges

    def test_fit_respects_skip_binary(self):

        train_df = pd.DataFrame({
            "binary": [0, 1, 0, 1],
            "numeric": [1, 2, 3, 4]
        })

        discretizer = Discretizer()
        discretizer.fit(train_df, method="equal_width", bins=2, skip_binary=True)


        assert 'binary' not in discretizer._bin_edges

        assert 'numeric' in discretizer._bin_edges

    def test_fit_with_specific_columns(self):

        train_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })

        discretizer = Discretizer()
        discretizer.fit(train_df, method="equal_width", bins=2, columns=["a"])


        assert 'a' in discretizer._bin_edges
        assert 'b' not in discretizer._bin_edges

    def test_transform_handles_out_of_range_values(self):

        train_df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        test_df = pd.DataFrame({"value": [5, 55]})

        discretizer = Discretizer()
        discretizer.fit(train_df, method="equal_width", bins=3)
        result = discretizer.transform(test_df)






        assert len(result) == 2

        train_result = discretizer.transform(train_df)
        assert all(val.startswith("bin_") for val in train_result["value"])

    def test_old_discretize_method_still_works(self):

        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

        discretizer = Discretizer()
        result = discretizer.discretize(df, method="equal_width", bins=2)


        assert len(result) == 5
        assert result["value"].nunique() == 2