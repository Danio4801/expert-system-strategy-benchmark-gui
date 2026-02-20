

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Set

from core.models import Fact, Rule


class RuleGenerator:









    

    ID_PATTERNS = ["id", "index", "row", "nr", "number", "lp", "name", "unnamed"]
    
    def __init__(self):

        self._statistics = {}
    
    def detect_id_columns(self, df: pd.DataFrame) -> List[str]:













        id_columns = []
        
        for col in df.columns:

            col_lower = col.lower()
            if any(pattern in col_lower for pattern in self.ID_PATTERNS):
                id_columns.append(col)
                continue
            

            if pd.api.types.is_numeric_dtype(df[col]):
                values = df[col].dropna()


                if len(values) == len(values.unique()):

                    sorted_vals = sorted(values)
                    if len(sorted_vals) > 1:
                        differences = [sorted_vals[i+1] - sorted_vals[i]
                                      for i in range(len(sorted_vals)-1)]
                        if all(d == 1 for d in differences):
                            id_columns.append(col)



            else:
                values = df[col].dropna()
                if len(values) >= 20:
                    unique_ratio = len(values.unique()) / len(values)
                    if unique_ratio > 0.95:
                        id_columns.append(col)
        
        return id_columns
    
    def generate(
        self, 
        df: pd.DataFrame, 
        decision_column: str,
        exclude_columns: Optional[List[str]] = None,
        auto_exclude_id: bool = False
    ) -> List[Rule]:

















        if len(df) == 0:
            raise ValueError("DataFrame jest pusty")
        

        if decision_column not in df.columns:
            raise ValueError(f"Kolumna decyzyjna '{decision_column}' nie istnieje w DataFrame")
        

        columns_to_exclude: Set[str] = set()
        
        if exclude_columns:
            columns_to_exclude.update(exclude_columns)
        
        if auto_exclude_id:
            id_cols = self.detect_id_columns(df)
            columns_to_exclude.update(id_cols)
        

        columns_to_exclude.add(decision_column)
        

        attribute_columns = [col for col in df.columns if col not in columns_to_exclude]
        

        if len(attribute_columns) == 0:
            raise ValueError("Brak kolumn atrybutów (wszystkie kolumny wykluczone)")
        

        relevant_columns = attribute_columns + [decision_column]
        df_unique = df[relevant_columns].drop_duplicates()
        

        rules = []
        for idx, row in df_unique.iterrows():
            premises = [Fact(col, str(row[col])) for col in attribute_columns]
            conclusion = Fact(decision_column, str(row[decision_column]))
            rule = Rule(id=len(rules), premises=premises, conclusion=conclusion)
            rules.append(rule)
        

        self._statistics = {
            "total_rules": len(rules),
            "avg_premises": sum(len(r.premises) for r in rules) / len(rules) if rules else 0,
            "excluded_columns": list(columns_to_exclude - {decision_column}),
            "attribute_columns": attribute_columns
        }
        
        return rules
    
    def get_statistics(self) -> Dict[str, Any]:






        return self._statistics