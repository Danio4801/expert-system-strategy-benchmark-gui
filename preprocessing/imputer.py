










import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal


@dataclass
class ImputationReport:









    total_missing: int
    columns_affected: List[str]
    values_imputed: Dict[str, int]
    method_used: str
    imputation_values: Dict[str, Dict[str, any]] = field(default_factory=dict)


class Imputer:
















    
    def __init__(self):

        self._last_report: Optional[ImputationReport] = None
    
    def check_missing(self, df: pd.DataFrame) -> Dict[str, int]:









        missing = df.isnull().sum()
        return {col: count for col, count in missing.items() if count > 0}
    
    def has_missing(self, df: pd.DataFrame) -> bool:









        return df.isnull().any().any()
    
    def impute(
        self,
        df: pd.DataFrame,
        decision_column: str,
        numeric_method: Literal["mean", "median"] = "mean",
        categorical_method: Literal["mode"] = "mode",
        columns: Optional[List[str]] = None
    ) -> tuple[pd.DataFrame, ImputationReport]:






















        if decision_column not in df.columns:
            raise ValueError(f"Kolumna decyzyjna '{decision_column}' nie istnieje w DataFrame")
        
        if df[decision_column].isnull().any():
            raise ValueError(
                f"Kolumna decyzyjna '{decision_column}' zawiera brakujące wartości. "
                "Usuń te wiersze przed imputacją."
            )
        

        result = df.copy()
        

        missing_info = self.check_missing(df)
        
        if not missing_info:

            report = ImputationReport(
                total_missing=0,
                columns_affected=[],
                values_imputed={},
                method_used=f"{numeric_method}/{categorical_method}",
                imputation_values={}
            )
            self._last_report = report
            return result, report
        

        if columns is not None:
            columns_to_impute = [col for col in columns if col in missing_info]
        else:

            columns_to_impute = [col for col in missing_info.keys() 
                                if col != decision_column]
        

        values_imputed = {}
        imputation_values = {}
        

        for col in columns_to_impute:
            col_imputation_values = {}
            

            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            if is_numeric:

                if numeric_method == "mean":
                    fill_values = df.groupby(decision_column)[col].transform('mean')
                else:
                    fill_values = df.groupby(decision_column)[col].transform('median')
                

                for cls in df[decision_column].unique():
                    class_data = df[df[decision_column] == cls][col]
                    if numeric_method == "mean":
                        col_imputation_values[str(cls)] = class_data.mean()
                    else:
                        col_imputation_values[str(cls)] = class_data.median()
                        
            else:

                def get_mode(x):
                    mode_result = x.mode()
                    if len(mode_result) > 0:
                        return mode_result.iloc[0]
                    return None
                
                fill_values = df.groupby(decision_column)[col].transform(
                    lambda x: x.fillna(get_mode(x))
                )
                

                for cls in df[decision_column].unique():
                    class_data = df[df[decision_column] == cls][col]
                    mode_val = class_data.mode()
                    if len(mode_val) > 0:
                        col_imputation_values[str(cls)] = mode_val.iloc[0]
            

            missing_count = result[col].isnull().sum()
            

            result[col] = result[col].fillna(fill_values)
            

            values_imputed[col] = missing_count
            imputation_values[col] = col_imputation_values
        

        report = ImputationReport(
            total_missing=sum(missing_info.values()),
            columns_affected=columns_to_impute,
            values_imputed=values_imputed,
            method_used=f"{numeric_method}/{categorical_method}",
            imputation_values=imputation_values
        )
        
        self._last_report = report
        return result, report
    
    def get_last_report(self) -> Optional[ImputationReport]:






        return self._last_report
    
    def drop_missing(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:















        if threshold is not None:

            min_count = int(len(df.columns) * threshold)
            return df.dropna(thresh=min_count)
        
        if columns is not None:
            return df.dropna(subset=columns)
        
        return df.dropna()
