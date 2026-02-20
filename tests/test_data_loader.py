
















import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os
import sys

from preprocessing.data_loader import DataLoader, CSVConfig
from preprocessing.validators import (
    ValidationError,
    ValidationResult,
    validate_file_path,
    validate_file_content,
    detect_csv_config
)




@pytest.fixture
def temp_dir():

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_csv(temp_dir):

    path = temp_dir / "valid.csv"
    path.write_text("name,age,city\nAlice,30,Warsaw\nBob,25,Krakow\n", encoding="utf-8")
    return path


@pytest.fixture
def valid_csv_semicolon(temp_dir):

    path = temp_dir / "valid_semicolon.csv"
    path.write_text("name;age;city\nAlice;30;Warsaw\nBob;25;Krakow\n", encoding="utf-8")
    return path


@pytest.fixture
def valid_csv_no_header(temp_dir):

    path = temp_dir / "no_header.csv"
    path.write_text("Alice,30,Warsaw\nBob,25,Krakow\n", encoding="utf-8")
    return path




class TestFilePathValidation:


    
    def test_f01_file_not_exists(self, temp_dir):

        path = temp_dir / "nonexistent.csv"
        result = validate_file_path(path)
        
        assert result.is_valid == False
        assert any(e.code == "F01" for e in result.errors)
    
    def test_f02_path_is_directory(self, temp_dir):

        result = validate_file_path(temp_dir)
        
        assert result.is_valid == False
        assert any(e.code == "F02" for e in result.errors)
    
    @pytest.mark.skipif(sys.platform == "win32", reason="os.chmod nie odbiera uprawnień odczytu na Windows")
    def test_f03_no_read_permission(self, temp_dir):

        path = temp_dir / "no_permission.csv"
        path.write_text("a,b\n1,2\n")
        os.chmod(path, 0o000)
        
        try:
            result = validate_file_path(path)
            assert result.is_valid == False
            assert any(e.code == "F03" for e in result.errors)
        finally:
            os.chmod(path, 0o644)
    
    def test_f04_no_extension(self, temp_dir):

        path = temp_dir / "datafile"
        path.write_text("a,b\n1,2\n")
        
        result = validate_file_path(path)
        
        assert result.is_valid == False
        assert any(e.code == "F04" for e in result.errors)
    
    def test_f05_invalid_extension(self, temp_dir):

        path = temp_dir / "data.xlsx"
        path.write_text("a,b\n1,2\n")
        
        result = validate_file_path(path)
        
        assert result.is_valid == False
        assert any(e.code == "F05" for e in result.errors)
    
    def test_f05_valid_extensions(self, temp_dir):

        for ext in [".csv", ".txt"]:
            path = temp_dir / f"data{ext}"
            path.write_text("a,b\n1,2\n")
            
            result = validate_file_path(path)
            
            assert not any(e.code == "F05" for e in result.errors)
    
    def test_f06_normalize_extension_case(self, temp_dir):

        for ext in [".CSV", ".CsV", ".Csv", ".TXT"]:
            path = temp_dir / f"data{ext}"
            path.write_text("a,b\n1,2\n")
            
            result = validate_file_path(path)
            

            assert not any(e.code == "F05" for e in result.errors)
    
    def test_f07_empty_file(self, temp_dir):

        path = temp_dir / "empty.csv"
        path.write_text("")
        
        result = validate_file_path(path)
        
        assert result.is_valid == False
        assert any(e.code == "F07" for e in result.errors)
    
    def test_valid_file_path(self, valid_csv):

        result = validate_file_path(valid_csv)
        
        assert result.is_valid == True
        assert len(result.errors) == 0




class TestFileContentValidation:


    
    def test_c01_only_whitespace(self, temp_dir):

        path = temp_dir / "whitespace.csv"
        path.write_text("   \n\t\n   \n")
        
        result = validate_file_content(path)
        
        assert result.is_valid == False
        assert any(e.code == "C01" for e in result.errors)
    
    def test_c02_invalid_encoding(self, temp_dir):

        path = temp_dir / "bad_encoding.csv"

        path.write_bytes(b"name,value\n\xff\xfe,123\n")
        
        result = validate_file_content(path, encoding="utf-8")
        
        assert result.is_valid == False
        assert any(e.code == "C02" for e in result.errors)
    
    def test_c03_single_column(self, temp_dir):

        path = temp_dir / "single_column.csv"
        path.write_text("value\n1\n2\n3\n")
        
        result = validate_file_content(path, separator=",")
        
        assert result.is_valid == False
        assert any(e.code == "C03" for e in result.errors)
    
    def test_c04_no_data_rows(self, temp_dir):

        path = temp_dir / "header_only.csv"
        path.write_text("name,age,city\n")
        
        result = validate_file_content(path, separator=",", has_header=True)
        
        assert result.is_valid == False
        assert any(e.code == "C04" for e in result.errors)
    
    def test_valid_content(self, valid_csv):

        result = validate_file_content(valid_csv, separator=",")
        
        assert result.is_valid == True




class TestHeaderValidation:

    
    def test_h02_duplicate_headers(self, temp_dir):

        path = temp_dir / "duplicate_headers.csv"
        path.write_text("name,age,name\nAlice,30,Doe\n")
        
        result = validate_file_content(path, separator=",", has_header=True)
        
        assert result.is_valid == False
        assert any(e.code == "H02" for e in result.errors)
    
    def test_h03_empty_header(self, temp_dir):

        path = temp_dir / "empty_header.csv"
        path.write_text("name,,city\nAlice,30,Warsaw\n")
        
        result = validate_file_content(path, separator=",", has_header=True)
        
        assert result.is_valid == False
        assert any(e.code == "H03" for e in result.errors)




class TestDataConsistencyValidation:

    
    def test_d01_inconsistent_columns(self, temp_dir):

        path = temp_dir / "inconsistent.csv"
        path.write_text("a,b,c\n1,2\n3,4,5,6\n")
        
        result = validate_file_content(path, separator=",", has_header=True)
        
        assert result.is_valid == False
        assert any(e.code == "D01" for e in result.errors)




class TestAutoDetection:


    
    def test_detect_comma_separator(self, valid_csv):

        config = detect_csv_config(valid_csv)
        
        assert config.separator == ","
    
    def test_detect_semicolon_separator(self, valid_csv_semicolon):

        config = detect_csv_config(valid_csv_semicolon)
        
        assert config.separator == ";"
    
    def test_detect_tab_separator(self, temp_dir):

        path = temp_dir / "tab.csv"
        path.write_text("name\tage\tcity\nAlice\t30\tWarsaw\n")
        
        config = detect_csv_config(path)
        
        assert config.separator == "\t"
    
    def test_detect_has_header_true(self, valid_csv):

        config = detect_csv_config(valid_csv)
        
        assert config.has_header == True
    
    def test_detect_has_header_false(self, valid_csv_no_header):

        config = detect_csv_config(valid_csv_no_header)
        

        assert config.has_header == False
    
    def test_detect_decimal_comma(self, temp_dir):

        path = temp_dir / "decimal_comma.csv"
        path.write_text("name;value\nA;3,14\nB;2,71\n")
        
        config = detect_csv_config(path)
        
        assert config.decimal == ","
        assert config.separator == ";"




class TestDataLoader:


    
    def test_load_valid_csv(self, valid_csv):

        loader = DataLoader()
        df = loader.load(valid_csv)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["name", "age", "city"]
    
    def test_load_with_custom_config(self, valid_csv_semicolon):

        loader = DataLoader()
        config = CSVConfig(separator=";")
        df = loader.load(valid_csv_semicolon, config=config)
        
        assert len(df.columns) == 3
    
    def test_load_with_autodetect(self, valid_csv_semicolon):

        loader = DataLoader()
        df = loader.load(valid_csv_semicolon, autodetect=True)
        
        assert len(df.columns) == 3
    
    def test_load_nonexistent_raises_error(self, temp_dir):

        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load(temp_dir / "nonexistent.csv")
    
    def test_load_invalid_content_raises_error(self, temp_dir):

        path = temp_dir / "single_col.csv"
        path.write_text("value\n1\n2\n")
        
        loader = DataLoader()
        
        with pytest.raises(ValueError):
            loader.load(path)
    
    def test_validate_returns_validation_result(self, valid_csv):

        loader = DataLoader()
        result = loader.validate(valid_csv)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid == True
    
    def test_validate_invalid_file(self, temp_dir):

        path = temp_dir / "empty.csv"
        path.write_text("")
        
        loader = DataLoader()
        result = loader.validate(path)
        
        assert result.is_valid == False
        assert len(result.errors) > 0