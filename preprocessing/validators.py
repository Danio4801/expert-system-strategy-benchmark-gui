

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple
from pathlib import Path
import os
import pandas as pd


@dataclass
class ValidationError:

    code: str
    message: str
    is_critical: bool


@dataclass
class ValidationResult:

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    infos: List[str] = field(default_factory=list)
    detected_config: Optional["CSVConfig"] = None


def validate_file_path(path: Path) -> ValidationResult:











    errors = []
    

    if not path.exists():
        errors.append(ValidationError(
            code="F01",
            message=f"Plik nie istnieje: {path}",
            is_critical=True
        ))
        return ValidationResult(is_valid=False, errors=errors)
    

    if path.is_dir():
        errors.append(ValidationError(
            code="F02",
            message=f"Ścieżka wskazuje na folder, nie plik: {path}",
            is_critical=True
        ))
        return ValidationResult(is_valid=False, errors=errors)
    

    if not os.access(path, os.R_OK):
        errors.append(ValidationError(
            code="F03",
            message=f"Brak uprawnień do odczytu pliku: {path}",
            is_critical=True
        ))
        return ValidationResult(is_valid=False, errors=errors)
    

    if not path.suffix:
        errors.append(ValidationError(
            code="F04",
            message=f"Plik nie ma rozszerzenia: {path}",
            is_critical=True
        ))
        return ValidationResult(is_valid=False, errors=errors)
    

    valid_extensions = {".csv", ".txt"}
    if path.suffix.lower() not in valid_extensions:
        errors.append(ValidationError(
            code="F05",
            message=f"Niepoprawne rozszerzenie pliku: {path.suffix}. Dozwolone: .csv, .txt",
            is_critical=True
        ))
        return ValidationResult(is_valid=False, errors=errors)
    

    if path.stat().st_size == 0:
        errors.append(ValidationError(
            code="F07",
            message=f"Plik jest pusty (0 bajtów): {path}",
            is_critical=True
        ))
        return ValidationResult(is_valid=False, errors=errors)
    
    return ValidationResult(is_valid=True, errors=[])


def validate_file_content(path: Path, separator: str = ",", 
                          has_header: bool = True,
                          encoding: str = "utf-8") -> ValidationResult:











    errors = []
    warnings = []
    

    try:
        content = path.read_text(encoding=encoding)
    except UnicodeDecodeError as e:
        errors.append(ValidationError(
            code="C02",
            message=f"Niepoprawne kodowanie pliku (oczekiwano {encoding}): {e}",
            is_critical=True
        ))
        return ValidationResult(is_valid=False, errors=errors)
    

    if not content.strip():
        errors.append(ValidationError(
            code="C01",
            message="Plik zawiera tylko białe znaki",
            is_critical=True
        ))
        return ValidationResult(is_valid=False, errors=errors)
    

    lines = content.strip().split('\n')
    

    if has_header and len(lines) <= 1:
        errors.append(ValidationError(
            code="C04",
            message="Plik zawiera tylko nagłówki, brak wierszy z danymi",
            is_critical=True
        ))
        return ValidationResult(is_valid=False, errors=errors)
    

    first_line = lines[0].split(separator)
    

    if len(first_line) < 2:
        errors.append(ValidationError(
            code="C03",
            message=f"Plik ma tylko jedną kolumnę (separator: '{separator}')",
            is_critical=True
        ))
        return ValidationResult(is_valid=False, errors=errors)
    

    if has_header:
        headers = first_line
        if len(headers) != len(set(headers)):
            duplicates = [h for h in headers if headers.count(h) > 1]
            errors.append(ValidationError(
                code="H02",
                message=f"Zduplikowane nazwy kolumn: {set(duplicates)}",
                is_critical=True
            ))
        

        if any(not h.strip() for h in headers):
            errors.append(ValidationError(
                code="H03",
                message="Plik zawiera puste nazwy kolumn",
                is_critical=True
            ))
    

    expected_cols = len(first_line)
    start_row = 1 if has_header else 0
    
    for i, line in enumerate(lines[start_row:], start=start_row):
        cols = line.split(separator)
        if len(cols) != expected_cols:
            errors.append(ValidationError(
                code="D01",
                message=f"Niespójna liczba kolumn w wierszu {i+1}: oczekiwano {expected_cols}, znaleziono {len(cols)}",
                is_critical=True
            ))
            break
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def detect_csv_config(path: Path) -> "CSVConfig":








    from preprocessing.data_loader import CSVConfig
    
    content = path.read_text(encoding="utf-8")
    first_line = content.split('\n')[0]
    

    possible_separators = [',', ';', '\t', '|']
    separator_counts = {sep: first_line.count(sep) for sep in possible_separators}
    separator = max(separator_counts, key=separator_counts.get)
    

    if separator_counts[separator] == 0:
        separator = ","
    




    first_row = first_line.split(separator)

    def looks_like_header(cell):
        cell = cell.strip()
        if not cell:
            return False

        first_char = cell[0]
        return first_char.isalpha()

    has_header = all(looks_like_header(cell) for cell in first_row if cell.strip())
    


    decimal = "."
    if separator == ";" and "," in content:

        decimal = ","
    
    return CSVConfig(
        separator=separator,
        decimal=decimal,
        has_header=has_header,
        encoding="utf-8"
    )

def validate_decision_column(
    df: pd.DataFrame,
    decision_column: Union[str, int],
    drop_missing: bool = True,
    min_rows: int = 10,
    imbalance_threshold: float = 0.9
) -> Tuple[pd.DataFrame, ValidationResult]:


























    errors = []
    warnings = []
    infos = []
    

    if isinstance(decision_column, int):

        if decision_column < 0 or decision_column >= len(df.columns):
            errors.append(ValidationError(
                code="DC01",
                message=f"Indeks kolumny decyzyjnej {decision_column} poza zakresem (0-{len(df.columns)-1})",
                is_critical=True
            ))
            return df, ValidationResult(is_valid=False, errors=errors, warnings=warnings, infos=infos)

        decision_column = df.columns[decision_column]
    else:

        if decision_column not in df.columns:
            errors.append(ValidationError(
                code="DC01",
                message=f"Kolumna decyzyjna '{decision_column}' nie istnieje. Dostępne: {list(df.columns)}",
                is_critical=True
            ))
            return df, ValidationResult(is_valid=False, errors=errors, warnings=warnings, infos=infos)
    

    null_count = df[decision_column].isnull().sum()
    if null_count > 0:
        if drop_missing:
            original_len = len(df)
            df = df.dropna(subset=[decision_column])
            dropped = original_len - len(df)
            infos.append(f"DF02: Usunięto {dropped} wierszy z pustą kolumną decyzyjną '{decision_column}'")
        else:
            errors.append(ValidationError(
                code="DC02",
                message=f"Kolumna decyzyjna '{decision_column}' ma {null_count} pustych wartości",
                is_critical=True
            ))
    

    if len(df) < min_rows:
        errors.append(ValidationError(
            code="DF01",
            message=f"Za mało wierszy po czyszczeniu: {len(df)} < {min_rows}",
            is_critical=True
        ))
    

    unique_classes = df[decision_column].nunique()
    if unique_classes < 2:
        errors.append(ValidationError(
            code="DC03",
            message=f"Kolumna decyzyjna '{decision_column}' ma tylko {unique_classes} unikalną klasę (wymagane >= 2)",
            is_critical=True
        ))
    

    if len(df) > 0 and unique_classes >= 2:
        class_counts = df[decision_column].value_counts(normalize=True)
        max_class_ratio = class_counts.max()
        if max_class_ratio > imbalance_threshold:
            dominant_class = class_counts.idxmax()
            warnings.append(f"DC04: Niezbalansowane klasy - '{dominant_class}' stanowi {max_class_ratio:.1%} danych (próg: {imbalance_threshold:.0%})")
    
    is_valid = len(errors) == 0
    return df, ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        infos=infos
    )
