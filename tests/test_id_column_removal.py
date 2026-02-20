


















import pytest
import pandas as pd
import tempfile
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from csv_loader import load_csv, CSVLoadError
from preprocessing.rule_generator import RuleGenerator




@pytest.fixture
def temp_dir():

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _write_csv(temp_dir, filename, content):

    path = temp_dir / filename
    path.write_text(content, encoding="utf-8")
    return str(path)




class TestDetectUnnamedPattern:


    def test_detect_unnamed_0(self):

        df = pd.DataFrame({
            "Unnamed: 0": [0, 1, 2],
            "color": ["red", "blue", "green"],
            "class": ["A", "B", "C"]
        })

        generator = RuleGenerator()
        id_columns = generator.detect_id_columns(df)

        assert "Unnamed: 0" in id_columns

    def test_detect_unnamed_1(self):

        df = pd.DataFrame({
            "Unnamed: 1": [10, 20, 30],
            "color": ["red", "blue", "green"],
            "class": ["A", "B", "C"]
        })

        generator = RuleGenerator()
        id_columns = generator.detect_id_columns(df)

        assert "Unnamed: 1" in id_columns

    def test_detect_unnamed_case_insensitive(self):

        df = pd.DataFrame({
            "UNNAMED_col": [1, 2, 3],
            "color": ["red", "blue", "green"],
            "class": ["A", "B", "C"]
        })

        generator = RuleGenerator()
        id_columns = generator.detect_id_columns(df)

        assert "UNNAMED_col" in id_columns




class TestLoadCsvRemovesIdColumns:


    def test_removes_id_column(self, temp_dir):

        path = _write_csv(temp_dir, "with_id.csv",
            "Id,color,size,class\n1,red,big,A\n2,blue,small,B\n"
            "3,green,big,A\n4,red,small,B\n5,blue,big,A\n"
            "6,green,small,B\n7,red,big,A\n8,blue,small,B\n"
            "9,green,big,A\n10,red,small,B\n")

        df, meta = load_csv(path)

        assert "Id" not in df.columns
        assert "Id" in meta['removed_id_columns']

    def test_removes_unnamed_0_column(self, temp_dir):

        path = _write_csv(temp_dir, "with_unnamed.csv",
            "Unnamed: 0,color,size,class\n0,red,big,A\n1,blue,small,B\n"
            "2,green,big,A\n3,red,small,B\n4,blue,big,A\n"
            "5,green,small,B\n6,red,big,A\n7,blue,small,B\n"
            "8,green,big,A\n9,red,small,B\n")

        df, meta = load_csv(path)

        assert "Unnamed: 0" not in df.columns
        assert "Unnamed: 0" in meta['removed_id_columns']

    def test_removes_sequential_numeric_column(self, temp_dir):

        path = _write_csv(temp_dir, "with_seq.csv",
            "row_nr,color,size,class\n1,red,big,A\n2,blue,small,B\n"
            "3,green,big,A\n4,red,small,B\n5,blue,big,A\n"
            "6,green,small,B\n7,red,big,A\n8,blue,small,B\n"
            "9,green,big,A\n10,red,small,B\n")

        df, meta = load_csv(path)

        assert "row_nr" not in df.columns
        assert "row_nr" in meta['removed_id_columns']

    def test_metadata_reports_removed_columns(self, temp_dir):

        path = _write_csv(temp_dir, "with_id2.csv",
            "Id,color,size,class\n1,red,big,A\n2,blue,small,B\n"
            "3,green,big,A\n4,red,small,B\n5,blue,big,A\n"
            "6,green,small,B\n7,red,big,A\n8,blue,small,B\n"
            "9,green,big,A\n10,red,small,B\n")

        df, meta = load_csv(path)

        assert 'removed_id_columns' in meta
        assert isinstance(meta['removed_id_columns'], list)
        assert len(meta['removed_id_columns']) > 0

    def test_no_id_columns_empty_list(self, temp_dir):

        path = _write_csv(temp_dir, "no_id.csv",
            "color,size,class\nred,big,A\nblue,small,B\n"
            "green,big,A\nred,small,B\nblue,big,A\n"
            "green,small,B\nred,big,A\nblue,small,B\n"
            "green,big,A\nred,small,B\n")

        df, meta = load_csv(path)

        assert meta['removed_id_columns'] == []

    def test_decision_column_not_removed_as_id(self, temp_dir):


        path = _write_csv(temp_dir, "id_in_decision.csv",
            "color,size,id_class\nred,big,A\nblue,small,B\n"
            "green,big,A\nred,small,B\nblue,big,A\n"
            "green,small,B\nred,big,A\nblue,small,B\n"
            "green,big,A\nred,small,B\n")

        df, meta = load_csv(path, decision_column_index=-1)


        assert "id_class" in df.columns
        assert "id_class" not in meta['removed_id_columns']

    def test_column_count_updated_after_removal(self, temp_dir):

        path = _write_csv(temp_dir, "count_test.csv",
            "Id,color,size,class\n1,red,big,A\n2,blue,small,B\n"
            "3,green,big,A\n4,red,small,B\n5,blue,big,A\n"
            "6,green,small,B\n7,red,big,A\n8,blue,small,B\n"
            "9,green,big,A\n10,red,small,B\n")

        df, meta = load_csv(path)


        assert meta['columns_total'] == 3
        assert len(df.columns) == 3

    def test_decision_column_index_recalculated(self, temp_dir):


        path = _write_csv(temp_dir, "idx_test.csv",
            "Id,color,size,class\n1,red,big,A\n2,blue,small,B\n"
            "3,green,big,A\n4,red,small,B\n5,blue,big,A\n"
            "6,green,small,B\n7,red,big,A\n8,blue,small,B\n"
            "9,green,big,A\n10,red,small,B\n")

        df, meta = load_csv(path, decision_column_index=-1)


        assert meta['decision_column_name'] == 'class'
        assert meta['decision_column_index'] == 2
        assert df.columns[meta['decision_column_index']] == 'class'

    def test_normal_numeric_column_not_removed(self, temp_dir):

        path = _write_csv(temp_dir, "normal_numeric.csv",
            "age,color,class\n25,red,A\n30,blue,B\n"
            "25,green,A\n40,red,B\n30,blue,A\n"
            "35,green,B\n25,red,A\n30,blue,B\n"
            "40,green,A\n35,red,B\n")

        df, meta = load_csv(path)

        assert "age" in df.columns
        assert meta['removed_id_columns'] == []
