

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd

from preprocessing.validators import (
    ValidationResult, 
    validate_file_path, 
    validate_file_content,
    detect_csv_config
)


@dataclass
class CSVConfig:

    separator: str = ","
    decimal: str = "."
    has_header: bool = True
    encoding: str = "utf-8"


class DataLoader:













    
    def validate(self, path: Path, config: Optional[CSVConfig] = None) -> ValidationResult:









        if isinstance(path, str):
            path = Path(path)
        

        path_result = validate_file_path(path)
        if not path_result.is_valid:
            return path_result
        

        if config is None:
            config = CSVConfig()
        
        content_result = validate_file_content(
            path, 
            separator=config.separator,
            has_header=config.has_header,
            encoding=config.encoding
        )
        
        return content_result
    
    def load(self, path: Path, config: Optional[CSVConfig] = None, 
             autodetect: bool = False) -> pd.DataFrame:














        if isinstance(path, str):
            path = Path(path)
        

        if autodetect:
            config = detect_csv_config(path)
        elif config is None:
            config = CSVConfig()
        

        validation_result = self.validate(path, config)
        
        if not validation_result.is_valid:

            for error in validation_result.errors:
                if error.code in ["F01"]:
                    raise FileNotFoundError(f"Plik nie istnieje: {path}")
                if error.is_critical:
                    raise ValueError(f"Błąd walidacji ({error.code}): {error.message}")
        

        try:
            df = pd.read_csv(
                path,
                sep=config.separator,
                decimal=config.decimal,
                header=0 if config.has_header else None,
                encoding=config.encoding
            )
            return df
        except Exception as e:
            raise ValueError(f"Błąd podczas wczytywania pliku: {e}")
