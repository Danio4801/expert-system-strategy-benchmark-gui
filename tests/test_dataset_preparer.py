













import pytest
import pandas as pd
import numpy as np
from preprocessing.dataset_preparer import PreparedDataset, DatasetPreparer


class TestPreparedDataset:



    def test_create_prepared_dataset(self):

        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = PreparedDataset(df=df, changes_log=["Change 1", "Change 2"])

        assert len(result.df) == 3
        assert len(result.changes_log) == 2

    def test_empty_changes_log(self):

        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = PreparedDataset(df=df)

        assert len(result.changes_log) == 0

    def test_print_summary_does_not_crash(self):

        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = PreparedDataset(df=df, changes_log=["Converted 'col1' to numeric"])

        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        result.print_summary()
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "DATASET PREPARATION SUMMARY" in output


class TestReplaceMissingMarkers:



    def test_replaces_question_mark(self):

        df = pd.DataFrame({
            'col1': ['1', '2', '?', '4'],
            'class': ['A', 'B', 'A', 'B']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert result.df['col1'].isnull().sum() == 1
        assert len([c for c in result.changes_log if "missing markers" in c.lower()]) == 1

    def test_replaces_multiple_markers(self):

        df = pd.DataFrame({
            'col1': ['1', 'NA', 'n/a', '?', ''],
            'class': ['A', 'B', 'A', 'B', 'A']
        })


        preparer = DatasetPreparer(remove_constant=False)
        result = preparer.prepare(df, 'class')


        assert result.df['col1'].isnull().sum() == 4

        assert pd.api.types.is_numeric_dtype(result.df['col1'])

    def test_custom_missing_markers(self):

        df = pd.DataFrame({
            'col1': ['1', 'MISSING', '3', 'UNKNOWN'],
            'class': ['A', 'B', 'A', 'B']
        })

        preparer = DatasetPreparer(missing_markers={'MISSING', 'UNKNOWN'})
        result = preparer.prepare(df, 'class')

        assert result.df['col1'].isnull().sum() == 2

    def test_no_missing_markers(self):

        df = pd.DataFrame({
            'col1': ['1', '2', '3'],
            'class': ['A', 'B', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert result.df['col1'].isnull().sum() == 0
        assert not any("missing markers" in c.lower() for c in result.changes_log)


class TestStripWhitespace:



    def test_strips_leading_trailing_whitespace(self):

        df = pd.DataFrame({
            'col1': ['  a  ', ' b', 'c '],
            'class': ['A', 'B', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')

        assert result.df['col1'].tolist() == ['a', 'b', 'c']
        assert any("stripped" in c.lower() for c in result.changes_log)

    def test_no_changes_when_no_whitespace(self):

        df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'class': ['A', 'B', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert not any("stripped" in c.lower() for c in result.changes_log)

    def test_can_disable_strip_whitespace(self):

        df = pd.DataFrame({
            'col1': ['  a  ', ' b', 'c '],
            'class': ['A', 'B', 'A']
        })

        preparer = DatasetPreparer(strip_whitespace=False)
        result = preparer.prepare(df, 'class')


        assert result.df['col1'].iloc[0] == '  a  '
        assert not any("stripped" in c.lower() for c in result.changes_log)


class TestConvertNumericAsString:




    def test_converts_numeric_string_to_numeric(self):

        df = pd.DataFrame({
            'col1': ['1', '2', '3', '4', '5'],
            'class': ['A', 'B', 'A', 'B', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert pd.api.types.is_numeric_dtype(result.df['col1'])
        assert any("converted" in c.lower() and "numeric" in c.lower() for c in result.changes_log)

    def test_converts_with_missing_markers(self):

        df = pd.DataFrame({
            'col1': ['1.0', '2.0', '?', '4.0', '5.0'],
            'class': ['A', 'B', 'A', 'B', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert pd.api.types.is_numeric_dtype(result.df['col1'])
        assert result.df['col1'].isnull().sum() == 1

    def test_respects_numeric_threshold(self):

        df = pd.DataFrame({

            'col1': ['1', '2', 'a', 'b', 'c'],
            'class': ['A', 'B', 'A', 'B', 'A']
        })

        preparer = DatasetPreparer(numeric_threshold=0.8)
        result = preparer.prepare(df, 'class')


        assert result.df['col1'].dtype == 'object'

    def test_converts_with_lower_threshold(self):

        df = pd.DataFrame({

            'col1': ['1', '2', '3', 'a', 'b'],
            'class': ['A', 'B', 'A', 'B', 'A']
        })

        preparer = DatasetPreparer(numeric_threshold=0.5)
        result = preparer.prepare(df, 'class')


        assert pd.api.types.is_numeric_dtype(result.df['col1'])

        assert result.df['col1'].isnull().sum() == 2

    def test_does_not_convert_true_categorical(self):

        df = pd.DataFrame({
            'col1': ['red', 'blue', 'green', 'yellow', 'red'],
            'class': ['A', 'B', 'A', 'B', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert result.df['col1'].dtype == 'object'
        assert not any("converted" in c.lower() and "col1" in c for c in result.changes_log)


class TestRemoveConstantColumns:



    def test_removes_constant_column(self):

        df = pd.DataFrame({
            'const1': [5, 5, 5, 5],
            'var1': [1, 2, 3, 4],
            'class': ['A', 'B', 'A', 'B']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert 'const1' not in result.df.columns
        assert 'var1' in result.df.columns
        assert any("removed" in c.lower() and "constant" in c.lower() for c in result.changes_log)

    def test_removes_multiple_constant_columns(self):

        df = pd.DataFrame({
            'const1': [5, 5, 5],
            'const2': ['a', 'a', 'a'],
            'var1': [1, 2, 3],
            'class': ['A', 'B', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert 'const1' not in result.df.columns
        assert 'const2' not in result.df.columns
        assert 'var1' in result.df.columns

    def test_does_not_remove_decision_column_even_if_constant(self):

        df = pd.DataFrame({
            'var1': [1, 2, 3],
            'class': ['A', 'A', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert 'class' in result.df.columns

    def test_can_disable_remove_constant(self):

        df = pd.DataFrame({
            'const1': [5, 5, 5],
            'var1': [1, 2, 3],
            'class': ['A', 'B', 'A']
        })

        preparer = DatasetPreparer(remove_constant=False)
        result = preparer.prepare(df, 'class')


        assert 'const1' in result.df.columns


class TestIntegration:



    def test_full_prepare_all_fixes(self):

        df = pd.DataFrame({
            'col1': ['  1  ', ' 2 ', '?'],
            'col2': [5, 5, 5],
            'col3': ['a', 'b', 'c'],
            'class': ['A', 'B', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert pd.api.types.is_numeric_dtype(result.df['col1'])
        assert result.df['col1'].isnull().sum() == 1


        assert 'col2' not in result.df.columns


        assert result.df['col3'].dtype == 'object'


        assert len(result.changes_log) >= 3

    def test_heart_disease_like_dataset(self):

        df = pd.DataFrame({
            'ca': ['0.0', '1.0', '2.0', '?', '3.0'] * 20,
            'thal': ['3.0', '7.0', '6.0', '?', '3.0'] * 20,
            'age': [50, 60, 55, 45, 70] * 20,
            'class': ['0', '1', '0', '1', '0'] * 20
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert pd.api.types.is_numeric_dtype(result.df['ca'])
        assert pd.api.types.is_numeric_dtype(result.df['thal'])


        assert result.df['ca'].isnull().sum() == 20
        assert result.df['thal'].isnull().sum() == 20


        assert pd.api.types.is_numeric_dtype(result.df['age'])

    def test_prepare_empty_changes_when_already_clean(self):

        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5] * 20,
            'num2': [6, 7, 8, 9, 10] * 20,
            'class': ['A', 'B', 'A', 'B', 'A'] * 20
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert len(result.changes_log) == 0

    def test_prepare_with_validation(self):

        df = pd.DataFrame({
            'col1': ['1', '2', '?', '4', '5'] * 20,
            'class': ['A', 'B', 'A', 'B', 'A'] * 20
        })

        preparer = DatasetPreparer()
        prepared, report = preparer.prepare_with_validation(df, 'class')


        assert isinstance(prepared, PreparedDataset)
        assert hasattr(report, 'score')
        assert hasattr(report, 'verdict')


        assert pd.api.types.is_numeric_dtype(prepared.df['col1'])

    def test_raises_error_when_decision_column_not_found(self):

        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        preparer = DatasetPreparer()

        with pytest.raises(ValueError, match="Decision column 'nonexistent' not found"):
            preparer.prepare(df, 'nonexistent')

    def test_preserves_original_dataframe(self):

        original_df = pd.DataFrame({
            'col1': ['1', '2', '?'],
            'class': ['A', 'B', 'A']
        })


        original_copy = original_df.copy()

        preparer = DatasetPreparer()
        result = preparer.prepare(original_df, 'class')


        pd.testing.assert_frame_equal(original_df, original_copy)


        assert not original_df.equals(result.df)


class TestEdgeCases:



    def test_all_nan_column(self):

        df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [1, 2, 3],
            'class': ['A', 'B', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert 'col1' not in result.df.columns
        assert any("removed" in c.lower() and "constant" in c.lower() for c in result.changes_log)

    def test_single_row_dataset(self):

        df = pd.DataFrame({
            'col1': ['1'],
            'class': ['A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert len(result.df) == 1

    def test_numeric_column_unchanged(self):

        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4.5, 5.5, 6.5],
            'class': ['A', 'B', 'A']
        })

        preparer = DatasetPreparer()
        result = preparer.prepare(df, 'class')


        assert result.df['col1'].tolist() == [1, 2, 3]
        assert result.df['col2'].tolist() == [4.5, 5.5, 6.5]
