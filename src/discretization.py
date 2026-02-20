




import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
from scipy import stats


class DiscretizationError(Exception):

    pass






class DistributionAnalyzer:



    
    @staticmethod
    def analyze(data: np.ndarray) -> dict:














        if len(data) == 0:
            raise DiscretizationError("Pusta tablica danych")
        
        data = np.asarray(data, dtype=float)
        

        data_clean = data[np.isfinite(data)]
        if len(data_clean) == 0:
            raise DiscretizationError("Brak poprawnych wartości (same NaN/inf)")
        

        min_val = float(np.min(data_clean))
        max_val = float(np.max(data_clean))
        mean_val = float(np.mean(data_clean))
        std_val = float(np.std(data_clean))
        
        skewness = float(stats.skew(data_clean)) if len(data_clean) > 2 else 0.0
        

        cv = (std_val / mean_val * 100) if mean_val != 0 else 0.0
        

        range_ratio = (max_val / min_val) if min_val > 0 else float('inf')
        

        Q1, Q3 = np.percentile(data_clean, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = int(np.sum((data_clean < lower_bound) | (data_clean > upper_bound)))
        
        statistics = {
            'count': len(data_clean),
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'skewness': skewness,
            'cv': cv,
            'range_ratio': range_ratio,
            'outliers_count': outliers_count,
            'Q1': float(Q1),
            'Q3': float(Q3),
            'IQR': float(IQR)
        }
        

        score_ew = 0
        score_ef = 0
        

        if abs(skewness) < 0.5:
            score_ew += 3
        else:
            score_ef += 3
        

        if cv < 20:
            score_ew += 2
        else:
            score_ef += 2
        

        if range_ratio < 5:
            score_ew += 2
        else:
            score_ef += 2
        

        if outliers_count == 0:
            score_ew += 1
        else:
            score_ef += 1
        

        total_score = score_ew + score_ef
        
        if score_ew > score_ef:
            method = 'equal_width'
            confidence = (score_ew / total_score) * 100
        elif score_ef > score_ew:
            method = 'equal_frequency'
            confidence = (score_ef / total_score) * 100
        else:

            method = 'equal_width'
            confidence = 50.0
        

        explanation = DistributionAnalyzer._generate_explanation(
            statistics, method, confidence
        )
        
        return {
            'statistics': statistics,
            'recommendation': method,
            'confidence': round(confidence, 1),
            'explanation': explanation
        }
    
    @staticmethod
    def _generate_explanation(stats: dict, method: str, confidence: float) -> str:

        
        reasons = []
        

        if abs(stats['skewness']) < 0.5:
            reasons.append("rozkład symetryczny")
        else:
            reasons.append("rozkład skośny")
        

        if stats['outliers_count'] > 0:
            reasons.append(f"{stats['outliers_count']} outlierów")
        else:
            reasons.append("brak outlierów")
        

        if stats['cv'] < 20:
            reasons.append("niskie rozproszenie")
        else:
            reasons.append("wysokie rozproszenie")
        
        reasons_text = ", ".join(reasons)
        
        if method == 'equal_width':
            return f"Rekomendacja Equal Width ({confidence:.0f}%): {reasons_text}."
        else:
            return f"Rekomendacja Equal Frequency ({confidence:.0f}%): {reasons_text}."






class EqualWidthDiscretizer:



    
    def __init__(self, n_bins: int = 3):




        if n_bins < 1:
            raise DiscretizationError(f"n_bins musi być >= 1, otrzymano: {n_bins}")
        
        self.n_bins = n_bins
        self.bin_edges_ = None
        self.is_fitted_ = False
    
    def fit(self, data: Union[np.ndarray, List]) -> 'EqualWidthDiscretizer':









        data = np.asarray(data, dtype=float)
        

        if len(data) == 0:
            raise DiscretizationError("Pusta tablica danych")
        
        data_clean = data[np.isfinite(data)]
        if len(data_clean) == 0:
            raise DiscretizationError("Brak poprawnych wartości")
        
        min_val = np.min(data_clean)
        max_val = np.max(data_clean)
        

        if min_val == max_val:
            self.bin_edges_ = [(min_val, max_val)]
            self.is_fitted_ = True
            return self
        

        bin_width = (max_val - min_val) / self.n_bins
        
        edges = []
        for i in range(self.n_bins):
            left = min_val + i * bin_width
            right = min_val + (i + 1) * bin_width
            edges.append((left, right))
        
        self.bin_edges_ = edges
        self.is_fitted_ = True
        
        return self
    
    def transform(self, data: Union[np.ndarray, List]) -> np.ndarray:









        if not self.is_fitted_:
            raise DiscretizationError("Discretizer nie jest wytrenowany. Wywołaj fit() najpierw.")
        
        data = np.asarray(data, dtype=float)
        
        if len(data) == 0:
            return np.array([])
        
        result = np.zeros(len(data), dtype=int)
        
        for i, value in enumerate(data):
            if not np.isfinite(value):
                result[i] = -1
                continue
            

            assigned = False
            for bin_idx, (left, right) in enumerate(self.bin_edges_):

                if bin_idx == len(self.bin_edges_) - 1:
                    if left <= value <= right:
                        result[i] = bin_idx
                        assigned = True
                        break
                else:

                    if left <= value < right:
                        result[i] = bin_idx
                        assigned = True
                        break
            

            if not assigned:
                if value < self.bin_edges_[0][0]:
                    result[i] = 0
                else:
                    result[i] = len(self.bin_edges_) - 1
        
        return result
    
    def fit_transform(self, data: Union[np.ndarray, List]) -> np.ndarray:

        return self.fit(data).transform(data)
    
    def get_bin_edges(self) -> List[Tuple[float, float]]:

        if not self.is_fitted_:
            raise DiscretizationError("Discretizer nie jest wytrenowany")
        return self.bin_edges_






class EqualFrequencyDiscretizer:



    
    def __init__(self, n_bins: int = 3):




        if n_bins < 1:
            raise DiscretizationError(f"n_bins musi być >= 1, otrzymano: {n_bins}")
        
        self.n_bins = n_bins
        self.bin_edges_ = None
        self.is_fitted_ = False
    
    def fit(self, data: Union[np.ndarray, List]) -> 'EqualFrequencyDiscretizer':









        data = np.asarray(data, dtype=float)
        

        if len(data) == 0:
            raise DiscretizationError("Pusta tablica danych")
        
        data_clean = data[np.isfinite(data)]
        if len(data_clean) == 0:
            raise DiscretizationError("Brak poprawnych wartości")
        

        if len(np.unique(data_clean)) == 1:
            val = data_clean[0]
            self.bin_edges_ = [(val, val)]
            self.is_fitted_ = True
            return self
        


        percentiles = np.linspace(0, 100, self.n_bins + 1)
        quantiles = np.percentile(data_clean, percentiles)
        

        edges = []
        for i in range(len(quantiles) - 1):
            left = quantiles[i]
            right = quantiles[i + 1]
            

            if left < right:
                edges.append((left, right))
        

        if len(edges) == 0:

            val = quantiles[0]
            self.bin_edges_ = [(val, val)]
        else:
            self.bin_edges_ = edges
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data: Union[np.ndarray, List]) -> np.ndarray:









        if not self.is_fitted_:
            raise DiscretizationError("Discretizer nie jest wytrenowany. Wywołaj fit() najpierw.")
        
        data = np.asarray(data, dtype=float)
        
        if len(data) == 0:
            return np.array([])
        
        result = np.zeros(len(data), dtype=int)
        
        for i, value in enumerate(data):
            if not np.isfinite(value):
                result[i] = -1
                continue
            

            assigned = False
            for bin_idx, (left, right) in enumerate(self.bin_edges_):

                if bin_idx == len(self.bin_edges_) - 1:
                    if left <= value <= right:
                        result[i] = bin_idx
                        assigned = True
                        break
                else:

                    if left <= value < right:
                        result[i] = bin_idx
                        assigned = True
                        break
            

            if not assigned:
                if value < self.bin_edges_[0][0]:
                    result[i] = 0
                else:
                    result[i] = len(self.bin_edges_) - 1
        
        return result
    
    def fit_transform(self, data: Union[np.ndarray, List]) -> np.ndarray:

        return self.fit(data).transform(data)
    
    def get_bin_edges(self) -> List[Tuple[float, float]]:

        if not self.is_fitted_:
            raise DiscretizationError("Discretizer nie jest wytrenowany")
        return self.bin_edges_
