




















import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_imputation import (
    ClassMeanImputer,
    ImputationError,
    load_and_impute_csv,
    print_imputation_report
)






@pytest.fixture
def sample_data_complete():

    return pd.DataFrame({
        'waga': [70.5, 85.2, 62.0, 95.5],
        'wzrost': [180, 175, 165, 180],
        'wiek': [25, 30, 22, 35],
        'diagnoza': ['zdrowy', 'nadwaga', 'zdrowy', 'nadwaga']
    })


@pytest.fixture
def sample_data_with_missing():

    return pd.DataFrame({
        'waga': [70.5, np.nan, 62.0, 95.5, 58.3, np.nan],
        'wzrost': [180, 175, np.nan, 180, 160, 185],
        'wiek': [25, 30, 22, np.nan, 28, 40],
        'diagnoza': ['zdrowy', 'nadwaga', 'zdrowy', 'nadwaga', 'zdrowy', 'nadwaga']
    })


@pytest.fixture
def sample_data_missing_decision():

    return pd.DataFrame({
        'waga': [70.5, 85.2, 62.0, 95.5, 58.3],
        'wzrost': [180, 175, 165, 180, 160],
        'wiek': [25, 30, 22, 35, 28],
        'diagnoza': ['zdrowy', np.nan, 'zdrowy', 'nadwaga', np.nan]
    })


@pytest.fixture
def sample_data_with_text():

    return pd.DataFrame({
        'waga': ['70.5', 'błąd', '62.0', '95.5'],
        'wzrost': [180, 175, 'tekst', 180],
        'wiek': [25, 30, 22, 35],
        'diagnoza': ['zdrowy', 'nadwaga', 'zdrowy', 'nadwaga']
    })


@pytest.fixture
def temp_csv_file():

    content = """waga,wzrost,wiek,diagnoza
70.5,180,25,zdrowy
85.2,175,30,nadwaga
,165,22,zdrowy
95.5,,35,nadwaga
58.3,160,NaN,zdrowy
102.0,185,40,nadwaga"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path


    if os.path.exists(temp_path):
        os.remove(temp_path)






def test_imputer_initialization():

    imputer = ClassMeanImputer(decision_column_index=-1)
    assert imputer.decision_column_index == -1
    assert imputer.is_fitted_ is False
    assert imputer.class_means_ == {}


def test_imputer_with_complete_data(sample_data_complete):

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, report = imputer.fit_transform(sample_data_complete, verbose=False)

    assert len(df_clean) == len(sample_data_complete)
    assert report['rows_removed_missing_decision'] == 0
    assert report['rows_final'] == 4
    assert len(report['columns_imputed']) == 0


def test_imputer_removes_missing_decision_rows(sample_data_missing_decision):

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, report = imputer.fit_transform(sample_data_missing_decision, verbose=False)


    assert len(df_clean) == 3
    assert report['rows_removed_missing_decision'] == 2
    assert report['rows_final'] == 3


    assert 'zdrowy' in df_clean['diagnoza'].values
    assert 'nadwaga' in df_clean['diagnoza'].values
    assert df_clean['diagnoza'].isna().sum() == 0


def test_imputer_fills_missing_with_class_mean(sample_data_with_missing):

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, report = imputer.fit_transform(sample_data_with_missing, verbose=False)


    assert len(df_clean) == 6
    assert report['rows_removed_missing_decision'] == 0


    assert df_clean['waga'].isna().sum() == 0
    assert df_clean['wzrost'].isna().sum() == 0
    assert df_clean['wiek'].isna().sum() == 0


    assert 'waga' in imputer.class_means_
    class_means_waga = imputer.class_means_['waga']


    expected_mean_zdrowy = (70.5 + 62.0 + 58.3) / 3
    assert np.isclose(class_means_waga['zdrowy'], expected_mean_zdrowy, atol=0.1)


    assert np.isclose(class_means_waga['nadwaga'], 95.5, atol=0.1)


def test_imputer_converts_text_to_nan(sample_data_with_text):

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, report = imputer.fit_transform(sample_data_with_text, verbose=False)


    assert df_clean['waga'].dtype == np.float64
    assert df_clean['wzrost'].dtype == np.float64


    assert df_clean['waga'].isna().sum() == 0
    assert df_clean['wzrost'].isna().sum() == 0


def test_imputer_decision_column_index():


    df = pd.DataFrame({
        'diagnoza': ['zdrowy', 'nadwaga', 'zdrowy', 'nadwaga'],
        'waga': [70.5, np.nan, 62.0, 95.5],
        'wzrost': [180, 175, 165, 180]
    })

    imputer = ClassMeanImputer(decision_column_index=0)
    df_clean, report = imputer.fit_transform(df, verbose=False)

    assert report['decision_column'] == 'diagnoza'
    assert len(df_clean) == 4
    assert df_clean['waga'].isna().sum() == 0






def test_imputer_empty_dataframe():

    df_empty = pd.DataFrame()
    imputer = ClassMeanImputer()

    with pytest.raises(ImputationError, match="DataFrame jest pusty"):
        imputer.fit_transform(df_empty, verbose=False)


def test_imputer_all_missing_decision():

    df = pd.DataFrame({
        'waga': [70.5, 85.2, 62.0],
        'wzrost': [180, 175, 165],
        'diagnoza': [np.nan, np.nan, np.nan]
    })

    imputer = ClassMeanImputer(decision_column_index=-1)

    with pytest.raises(ImputationError, match="Wszystkie wiersze zostały usunięte"):
        imputer.fit_transform(df, verbose=False)


def test_imputer_single_class():

    df = pd.DataFrame({
        'waga': [70.5, np.nan, 62.0, np.nan],
        'wzrost': [180, 175, 165, 180],
        'diagnoza': ['zdrowy', 'zdrowy', 'zdrowy', 'zdrowy']
    })

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, report = imputer.fit_transform(df, verbose=False)


    expected_mean = (70.5 + 62.0) / 2
    assert np.isclose(imputer.class_means_['waga']['zdrowy'], expected_mean, atol=0.1)


    assert df_clean['waga'].isna().sum() == 0


def test_imputer_all_missing_in_column_per_class():

    df = pd.DataFrame({
        'waga': [70.5, np.nan, 62.0, np.nan],
        'wzrost': [180, 175, 165, 180],
        'diagnoza': ['zdrowy', 'nadwaga', 'zdrowy', 'nadwaga']
    })

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, report = imputer.fit_transform(df, verbose=False)



    assert 'waga' in imputer.class_means_
    assert 'nadwaga' in imputer.class_means_['waga']


    assert df_clean['waga'].isna().sum() == 0


def test_imputer_invalid_decision_column_index():

    df = pd.DataFrame({
        'waga': [70.5, 85.2],
        'diagnoza': ['zdrowy', 'nadwaga']
    })


    imputer = ClassMeanImputer(decision_column_index=10)

    with pytest.raises(ImputationError, match="Nieprawidłowy indeks kolumny decyzyjnej"):
        imputer.fit_transform(df, verbose=False)


def test_imputer_no_numeric_columns():

    df = pd.DataFrame({
        'kolor': ['czerwony', 'niebieski', 'zielony'],
        'kształt': ['okrągły', 'kwadratowy', 'trójkąt'],
        'diagnoza': ['A', 'B', 'A']
    })

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, report = imputer.fit_transform(df, verbose=False)



    assert len(df_clean) == 3

    assert 'kolor' in report['columns_imputed']
    assert 'kształt' in report['columns_imputed']






def test_load_and_impute_csv_success(temp_csv_file):

    df_clean, report = load_and_impute_csv(
        temp_csv_file,
        decision_column_index=-1,
        verbose=False
    )

    assert report['filename'] == Path(temp_csv_file).name
    assert report['decision_column'] == 'diagnoza'
    assert report['rows_final'] > 0
    assert len(df_clean) == report['rows_final']


    assert df_clean['waga'].isna().sum() == 0
    assert df_clean['wzrost'].isna().sum() == 0
    assert df_clean['wiek'].isna().sum() == 0


def test_load_and_impute_csv_file_not_found():

    with pytest.raises(ImputationError, match="Plik nie istnieje"):
        load_and_impute_csv("nieistniejący_plik.csv", verbose=False)


def test_get_class_means_before_fit():

    imputer = ClassMeanImputer()

    with pytest.raises(ImputationError, match="Imputer nie jest dopasowany"):
        imputer.get_class_means()


def test_get_class_means_after_fit(sample_data_with_missing):

    imputer = ClassMeanImputer(decision_column_index=-1)
    imputer.fit_transform(sample_data_with_missing, verbose=False)

    class_means = imputer.get_class_means()

    assert isinstance(class_means, dict)
    assert 'waga' in class_means
    assert 'zdrowy' in class_means['waga']
    assert 'nadwaga' in class_means['waga']


def test_print_imputation_report(sample_data_with_missing, capsys):

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, report = imputer.fit_transform(sample_data_with_missing, verbose=False)

    print_imputation_report(report)

    captured = capsys.readouterr()
    assert "RAPORT IMPUTACJI DANYCH" in captured.out
    assert "diagnoza" in captured.out
    assert "Wiersze:" in captured.out






def test_full_workflow_with_real_data():


    df = pd.DataFrame({
        'waga': [70.5, 'błąd', 62.0, 95.5, np.nan, 102.0, 68.0, np.nan],
        'wzrost': [180, 175, np.nan, 180, 160, 185, 170, 178],
        'wiek': [25, 30, 22, np.nan, 28, 40, 26, 32],
        'bmi': [21.8, 27.8, 22.8, 29.5, np.nan, 29.8, 23.5, 24.8],
        'diagnoza': ['zdrowy', 'nadwaga', 'zdrowy', 'nadwaga', 'zdrowy', np.nan, 'zdrowy', 'nadwaga']
    })

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, report = imputer.fit_transform(df, verbose=False)



    assert len(df_clean) == 7


    assert df_clean['waga'].isna().sum() == 0
    assert df_clean['wzrost'].isna().sum() == 0
    assert df_clean['wiek'].isna().sum() == 0
    assert df_clean['bmi'].isna().sum() == 0


    assert report['rows_removed_missing_decision'] == 1
    assert report['rows_final'] == 7
    assert len(report['columns_imputed']) == 4


    assert set(df_clean['diagnoza'].unique()) == {'zdrowy', 'nadwaga'}


def test_different_separators():

    content = "waga;wzrost;wiek;diagnoza\n70,5;180;25;zdrowy\n85,2;175;30;nadwaga"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    try:
        df_clean, report = load_and_impute_csv(
            temp_path,
            column_separator=';',
            decimal_separator=',',
            verbose=False
        )

        assert len(df_clean) == 2
        assert df_clean['waga'].dtype == np.float64
        assert np.isclose(df_clean['waga'].iloc[0], 70.5, atol=0.1)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)






def test_report_structure(sample_data_with_missing):

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, report = imputer.fit_transform(sample_data_with_missing, verbose=False)


    required_keys = [
        'decision_column',
        'rows_original',
        'rows_removed_missing_decision',
        'rows_final',
        'columns_imputed',
        'imputation_summary'
    ]

    for key in required_keys:
        assert key in report, f"Brak klucza '{key}' w raporcie"


    assert isinstance(report['decision_column'], str)
    assert isinstance(report['rows_original'], int)
    assert isinstance(report['rows_final'], int)
    assert isinstance(report['columns_imputed'], dict)
    assert isinstance(report['imputation_summary'], list)


def test_imputation_preserves_data_integrity():

    df = pd.DataFrame({
        'waga': [70.5, np.nan, 62.0, 95.5],
        'wzrost': [180, 175, 165, 180],
        'diagnoza': ['zdrowy', 'nadwaga', 'zdrowy', 'nadwaga']
    })

    df_original = df.copy()

    imputer = ClassMeanImputer(decision_column_index=-1)
    df_clean, _ = imputer.fit_transform(df, verbose=False)


    assert df_clean.loc[0, 'waga'] == df_original.loc[0, 'waga']
    assert df_clean.loc[2, 'waga'] == df_original.loc[2, 'waga']
    assert df_clean.loc[3, 'waga'] == df_original.loc[3, 'waga']


    assert not np.isnan(df_clean.loc[1, 'waga'])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
