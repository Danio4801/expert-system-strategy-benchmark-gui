




import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path


class ImputationError(Exception):

    pass


class ClassMeanImputer:










    def __init__(self, decision_column_index: int = -1):




        self.decision_column_index = decision_column_index
        self.class_means_ = {}
        self.is_fitted_ = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:










        if df.empty:
            raise ImputationError("DataFrame jest pusty")

        df_clean = df.copy()


        if self.decision_column_index == -1:
            dec_col_idx = len(df_clean.columns) - 1
        else:
            dec_col_idx = self.decision_column_index

        if dec_col_idx < 0 or dec_col_idx >= len(df_clean.columns):
            raise ImputationError(
                f"Nieprawidłowy indeks kolumny decyzyjnej: {self.decision_column_index}"
            )

        decision_column = df_clean.columns[dec_col_idx]


        report = {
            'decision_column': decision_column,
            'rows_original': len(df_clean),
            'rows_removed_missing_decision': 0,
            'rows_final': 0,
            'columns_imputed': {},
            'imputation_summary': []
        }


        rows_before = len(df_clean)
        df_clean = df_clean[df_clean[decision_column].notna()]
        rows_after = len(df_clean)
        rows_removed = rows_before - rows_after

        report['rows_removed_missing_decision'] = rows_removed

        if verbose and rows_removed > 0:
            print(f"[KROK 1] Usunięto {rows_removed} wierszy z brakującą wartością decyzyjną")

        if df_clean.empty:
            raise ImputationError(
                "Wszystkie wiersze zostały usunięte (brak poprawnych wartości decyzyjnych)"
            )


        classes = df_clean[decision_column].unique()

        if verbose:
            print(f"[INFO] Znaleziono {len(classes)} klas decyzyjnych: {list(classes)}")


        numeric_columns = []
        for col_idx, col in enumerate(df_clean.columns):
            if col_idx == dec_col_idx:
                continue



            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                numeric_columns.append(col)
            except Exception:
                pass

        if verbose:
            print(f"[INFO] Kolumny numeryczne do imputacji: {numeric_columns}")


        self.class_means_ = {}

        for col in numeric_columns:
            missing_count_before = df_clean[col].isna().sum()

            if missing_count_before == 0:
                if verbose:
                    print(f"[{col}] Brak wartości do imputacji")
                continue


            class_means_for_col = {}

            for cls in classes:

                mask = df_clean[decision_column] == cls
                class_data = df_clean.loc[mask, col]


                valid_values = class_data.dropna()

                if len(valid_values) > 0:
                    mean_value = valid_values.mean()
                    class_means_for_col[cls] = mean_value
                else:

                    global_mean = df_clean[col].mean()
                    class_means_for_col[cls] = global_mean if not np.isnan(global_mean) else 0.0

            self.class_means_[col] = class_means_for_col


            for cls in classes:
                mask = (df_clean[decision_column] == cls) & (df_clean[col].isna())
                num_imputed = mask.sum()

                if num_imputed > 0:
                    mean_val = class_means_for_col[cls]
                    df_clean.loc[mask, col] = mean_val

                    if verbose:
                        print(f"[{col}] Klasa '{cls}': Uzupelniono {num_imputed} wartosci -> {mean_val:.2f}")

            missing_count_after = df_clean[col].isna().sum()

            report['columns_imputed'][col] = {
                'missing_before': int(missing_count_before),
                'missing_after': int(missing_count_after),
                'imputed_count': int(missing_count_before - missing_count_after),
                'class_means': {k: float(v) for k, v in class_means_for_col.items()}
            }

            report['imputation_summary'].append(
                f"{col}: {missing_count_before} -> {missing_count_after} (uzupelniono: {missing_count_before - missing_count_after})"
            )

        report['rows_final'] = len(df_clean)
        self.is_fitted_ = True

        if verbose:
            print(f"\n[WYNIK] Wiersze: {report['rows_original']} -> {report['rows_final']}")
            print(f"[WYNIK] Usunieto: {report['rows_removed_missing_decision']} wierszy")
            print(f"[WYNIK] Zaimputowano dane w {len(report['columns_imputed'])} kolumnach")

        return df_clean, report

    def get_class_means(self) -> Dict:






        if not self.is_fitted_:
            raise ImputationError("Imputer nie jest dopasowany. Wywołaj fit_transform() najpierw.")

        return self.class_means_


def load_and_impute_csv(
    filepath: str,
    decision_column_index: int = -1,
    column_separator: str = ',',
    decimal_separator: str = '.',
    has_header: bool = True,
    encoding: str = 'utf-8',
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:


















    filepath_obj = Path(filepath)


    if not filepath_obj.exists():
        raise ImputationError(f"Plik nie istnieje: {filepath}")

    if not filepath_obj.is_file():
        raise ImputationError(f"Podana ścieżka nie jest plikiem: {filepath}")

    try:

        df = pd.read_csv(
            filepath,
            sep=column_separator,
            decimal=decimal_separator,
            header=0 if has_header else None,
            encoding=encoding,
            skipinitialspace=True,
            keep_default_na=True
        )
    except Exception as e:
        raise ImputationError(f"Błąd wczytywania pliku: {str(e)}")

    if df.empty:
        raise ImputationError("Plik CSV jest pusty")

    if verbose:
        print(f"[LOAD] Wczytano plik: {filepath_obj.name}")
        print(f"[LOAD] Rozmiar: {len(df)} wierszy, {len(df.columns)} kolumn")


    imputer = ClassMeanImputer(decision_column_index=decision_column_index)
    df_clean, report = imputer.fit_transform(df, verbose=verbose)


    report['filepath'] = str(filepath)
    report['filename'] = filepath_obj.name
    report['columns'] = list(df_clean.columns)

    return df_clean, report


def print_imputation_report(report: Dict) -> None:






    print("\n" + "="*70)
    print("RAPORT IMPUTACJI DANYCH")
    print("="*70)

    if 'filename' in report:
        print(f"Plik: {report['filename']}")

    print(f"Kolumna decyzyjna: '{report['decision_column']}'")
    print(f"\nWiersze:")
    print(f"  Początkowe:      {report['rows_original']}")
    print(f"  Usunięte:        {report['rows_removed_missing_decision']} (brak wartości decyzyjnej)")
    print(f"  Końcowe:         {report['rows_final']}")

    if report['columns_imputed']:
        print(f"\nImputacja wartości:")
        for col, info in report['columns_imputed'].items():
            print(f"\n  [{col}]")
            print(f"    Brakujące przed: {info['missing_before']}")
            print(f"    Brakujące po:    {info['missing_after']}")
            print(f"    Uzupełniono:     {info['imputed_count']}")
            print(f"    Średnie per klasa:")
            for cls, mean_val in info['class_means'].items():
                print(f"      {cls}: {mean_val:.2f}")
    else:
        print("\nBrak wartości do imputacji")

    print("="*70 + "\n")
