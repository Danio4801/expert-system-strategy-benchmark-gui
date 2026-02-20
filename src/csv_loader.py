



import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import os
import sys

if hasattr(sys, '_MEIPASS'):
    sys.path.insert(0, sys._MEIPASS)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.rule_generator import RuleGenerator


class CSVLoadError(Exception):

    pass


def load_csv(
    filepath: str,
    column_separator: str = ',',
    decimal_separator: str = '.',
    has_header: bool = True,
    decision_column_index: int = -1,
    drop_missing: bool = True,
    encoding: str = 'utf-8'
) -> Tuple[pd.DataFrame, Dict]:






















    

    filepath_obj = Path(filepath)
    
    if not filepath_obj.exists():
        raise CSVLoadError(f"Plik nie istnieje: {filepath}")
    
    if not filepath_obj.is_file():
        raise CSVLoadError(f"Podana ścieżka nie jest plikiem: {filepath}")
    

    if not os.access(filepath, os.R_OK):
        raise CSVLoadError(f"Brak uprawnień do odczytu pliku: {filepath}")
    

    valid_extensions = ['.csv', '.txt', '.CSV', '.TXT']
    if filepath_obj.suffix not in valid_extensions:
        raise CSVLoadError(
            f"Nieprawidłowe rozszerzenie pliku: {filepath_obj.suffix}. "
            f"Dozwolone: {', '.join(valid_extensions)}"
        )
    

    file_size = filepath_obj.stat().st_size
    if file_size == 0:
        raise CSVLoadError("Plik jest pusty (0 bajtów)")
    

    max_size_mb = 100
    if file_size > max_size_mb * 1024 * 1024:
        print(f"!  Ostrzezenie: Plik jest duzy ({file_size / 1024 / 1024:.1f} MB). "
              f"Wczytywanie moze byc wolne.")
    

    try:
        df = pd.read_csv(
            filepath,
            sep=column_separator,
            decimal=decimal_separator,
            header=0 if has_header else None,
            encoding=encoding,
            skipinitialspace=True,
            na_values=['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'nan', 'NaN', 'None'],
            keep_default_na=True
        )
    except UnicodeDecodeError:

        try:
            df = pd.read_csv(
                filepath,
                sep=column_separator,
                decimal=decimal_separator,
                header=0 if has_header else None,
                encoding='windows-1250',
                skipinitialspace=True,
                na_values=['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'nan', 'NaN', 'None'],
                keep_default_na=True
            )
            print(f"*  Plik wczytano z kodowaniem windows-1250")
        except Exception as e:
            raise CSVLoadError(f"Błąd kodowania pliku: {str(e)}")
    except pd.errors.ParserError as e:
        raise CSVLoadError(f"Błąd parsowania CSV: {str(e)}")
    except Exception as e:
        raise CSVLoadError(f"Błąd wczytywania pliku: {str(e)}")
    

    if df.empty:
        raise CSVLoadError("Plik nie zawiera żadnych danych")
    

    if not has_header:
        df.columns = [f'kolumna_{i+1}' for i in range(len(df.columns))]
    

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise CSVLoadError(f"Wykryto duplikaty w nazwach kolumn: {duplicates}")
    

    if df.columns.isnull().any():
        raise CSVLoadError("Niektóre kolumny nie mają nazw")
    

    if len(df.columns) < 2:
        raise CSVLoadError(
            f"Za mało kolumn: {len(df.columns)}. "
            f"Wymagane minimum: 2 (przynajmniej 1 warunkowa + 1 decyzyjna)"
        )
    


    if (df.isnull().all(axis=1)).any():
        empty_rows = df[df.isnull().all(axis=1)].index.tolist()
        print(f"!  Ostrzezenie: Wykryto puste wiersze: {empty_rows}")
        df = df[~df.isnull().all(axis=1)]




    _detector = RuleGenerator()
    detected_id_columns = _detector.detect_id_columns(df)


    effective_dec_idx = decision_column_index if decision_column_index >= 0 else len(df.columns) - 1
    decision_col_name_before_drop = df.columns[effective_dec_idx]


    detected_id_columns = [c for c in detected_id_columns if c != decision_col_name_before_drop]

    removed_id_columns = []
    if detected_id_columns:
        removed_id_columns = list(detected_id_columns)
        df = df.drop(columns=detected_id_columns)
        print(f"*  Automatycznie usunięto kolumny indeksowe: {detected_id_columns}")


        decision_column_index = df.columns.get_loc(decision_col_name_before_drop)


        if len(df.columns) < 2:
            raise CSVLoadError(
                f"Po usunięciu kolumn indeksowych ({detected_id_columns}) zostało za mało kolumn: "
                f"{len(df.columns)}. Wymagane minimum: 2 (1 warunkowa + 1 decyzyjna)"
            )


    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    missing_report = {
        col: {'count': int(missing_count[col]), 'percent': float(missing_percent[col])}
        for col in df.columns
    }
    
    total_missing_rows = df.isnull().any(axis=1).sum()
    

    if decision_column_index < -1 or decision_column_index >= len(df.columns):
        raise CSVLoadError(
            f"Nieprawidłowy indeks kolumny decyzyjnej: {decision_column_index}. "
            f"Dostępne kolumny: 0 do {len(df.columns)-1} (lub -1 dla ostatniej)"
        )
    

    if decision_column_index == -1:
        decision_column_index = len(df.columns) - 1
    
    decision_column_name = df.columns[decision_column_index]
    

    decision_col = df.iloc[:, decision_column_index]
    if decision_col.isnull().all():
        raise CSVLoadError(f"Kolumna decyzyjna '{decision_column_name}' jest całkowicie pusta")
    
    if decision_col.isnull().any():
        nan_count = decision_col.isnull().sum()
        if drop_missing:
            print(f"!  Kolumna decyzyjna '{decision_column_name}' ma {nan_count} pustych wartosci. "
                  f"Wiersze zostana usuniete.")
        else:
            raise CSVLoadError(
                f"Kolumna decyzyjna '{decision_column_name}' ma {nan_count} pustych wartości. "
                f"Nie można trenować bez znanych etykiet. Włącz 'drop_missing' lub napraw dane."
            )
    

    unique_classes = decision_col.dropna().nunique()
    if unique_classes < 2:
        raise CSVLoadError(
            f"Kolumna decyzyjna '{decision_column_name}' ma tylko {unique_classes} unikalną wartość. "
            f"System ekspertowy wymaga przynajmniej 2 klas."
        )
    

    rows_before = len(df)
    if drop_missing:
        df = df.dropna()
        rows_after = len(df)
        dropped_rows = rows_before - rows_after
        
        if dropped_rows > 0:
            print(f"*  Usunieto {dropped_rows} wierszy z wartosciami pustymi "
                  f"({dropped_rows/rows_before*100:.1f}%)")
    else:
        dropped_rows = 0
    

    min_rows = 10
    if len(df) < min_rows:
        raise CSVLoadError(
            f"Za mało danych po walidacji: {len(df)} wierszy (minimum: {min_rows})"
        )
    

    class_distribution = decision_col.value_counts(normalize=True) * 100
    max_class_percent = class_distribution.max()
    
    if max_class_percent > 90:
        print(f"!  Ostrzezenie: Niezbalansowane klasy. "
              f"Dominujaca klasa: {max_class_percent:.1f}%")
    

    metadata = {
        'filepath': str(filepath),
        'filename': filepath_obj.name,
        'file_size_bytes': file_size,
        'encoding': encoding,
        'column_separator': column_separator,
        'decimal_separator': decimal_separator,
        'has_header': has_header,
        'rows_original': rows_before,
        'rows_final': len(df),
        'dropped_rows': dropped_rows,
        'columns_total': len(df.columns),
        'column_names': df.columns.tolist(),
        'decision_column_index': decision_column_index,
        'decision_column_name': decision_column_name,
        'decision_classes': sorted(df[decision_column_name].unique().tolist()),
        'decision_classes_count': unique_classes,
        'class_distribution': class_distribution.to_dict(),
        'missing_values': missing_report,
        'total_missing_rows': int(total_missing_rows),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'removed_id_columns': removed_id_columns
    }
    
    return df, metadata


def print_metadata(metadata: Dict) -> None:






    print("\n" + "="*60)
    print("RAPORT WCZYTANIA DANYCH")
    print("="*60)
    print(f"Plik: {metadata['filename']}")
    print(f"Rozmiar: {metadata['file_size_bytes'] / 1024:.1f} KB")
    print(f"Kodowanie: {metadata['encoding']}")
    print(f"Separatory: kolumny='{metadata['column_separator']}', "
          f"dziesiętny='{metadata['decimal_separator']}'")
    print(f"\nWiersze: {metadata['rows_final']} "
          f"(usunięto {metadata['dropped_rows']})")
    print(f"Kolumny: {metadata['columns_total']}")
    print(f"  {', '.join(metadata['column_names'])}")
    print(f"\nKolumna decyzyjna: '{metadata['decision_column_name']}'")
    print(f"  Liczba klas: {metadata['decision_classes_count']}")
    print(f"  Klasy: {metadata['decision_classes']}")
    print(f"\nRozkład klas:")
    for cls, pct in metadata['class_distribution'].items():
        print(f"  {cls}: {pct:.1f}%")
    
    if metadata['total_missing_rows'] > 0:
        print(f"\n!  Brakujace wartosci w {metadata['total_missing_rows']} wierszach")
    
    print("="*60 + "\n")