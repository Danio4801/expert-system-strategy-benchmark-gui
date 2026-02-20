

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.cluster import KMeans


class Discretizer:



















    def __init__(self):

        self._bin_edges = {}
        self._method = None
        self._bins = None
        self._columns = None
        self._skip_binary = True
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        method: str = "equal_width",
        bins: int = 5,
        columns: Optional[List[str]] = None,
        skip_binary: bool = True
    ) -> "Discretizer":



















        if bins <= 0:
            raise ValueError("Liczba binów musi być większa od 0")
        if len(df) == 0:
            raise ValueError("DataFrame jest pusty")

        valid_methods = ["equal_width", "equal_frequency", "kmeans"]
        if method not in valid_methods:
            raise ValueError(f"Niepoprawna metoda: {method}. Dozwolone: {valid_methods}")


        self._method = method
        self._bins = bins
        self._skip_binary = skip_binary


        if columns is None:
            self._columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self._columns = columns


        self._bin_edges = {}
        for column in self._columns:

            if skip_binary and df[column].nunique() <= 2:
                continue


            if method == "equal_width":
                self._bin_edges[column] = self._fit_equal_width(df[column], bins)
            elif method == "equal_frequency":
                self._bin_edges[column] = self._fit_equal_frequency(df[column], bins)
            elif method == "kmeans":
                self._bin_edges[column] = self._fit_kmeans(df[column], bins)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:












        if not self._fitted:
            raise ValueError("Discretizer nie został dopasowany. Użyj fit() przed transform().")

        result = df.copy()

        for column in self._bin_edges.keys():
            if column not in df.columns:
                continue

            edge_info = self._bin_edges[column]

            if self._method == "equal_width":
                result[column] = self._transform_equal_width(df[column], edge_info)
            elif self._method == "equal_frequency":
                result[column] = self._transform_equal_frequency(df[column], edge_info)
            elif self._method == "kmeans":
                result[column] = self._transform_kmeans(df[column], edge_info)

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        method: str = "equal_width",
        bins: int = 5,
        columns: Optional[List[str]] = None,
        skip_binary: bool = True
    ) -> pd.DataFrame:















        self.fit(df, method=method, bins=bins, columns=columns, skip_binary=skip_binary)
        return self.transform(df)

    def discretize(
        self,
        df: pd.DataFrame,
        method: str = "equal_width",
        bins: int = 5,
        columns: Optional[List[str]] = None,
        skip_binary: bool = True
    ) -> pd.DataFrame:




















        discretizer = Discretizer()
        return discretizer.fit_transform(
            df, method=method, bins=bins, columns=columns, skip_binary=skip_binary
        )
    


    def _fit_equal_width(self, series: pd.Series, bins: int) -> dict:

        _, bin_edges = pd.cut(series, bins=bins, retbins=True)
        return {"bins": bins, "edges": bin_edges}

    def _fit_equal_frequency(self, series: pd.Series, bins: int) -> dict:

        try:
            _, bin_edges = pd.qcut(series, q=bins, retbins=True, duplicates="drop")
        except ValueError:

            result = pd.qcut(series, q=bins, duplicates="drop", retbins=False)
            bin_edges = [result.min().left] + [cat.right for cat in result.cat.categories]
            bin_edges = np.array(bin_edges)
        return {"bins": bins, "edges": bin_edges}

    def _fit_kmeans(self, series: pd.Series, bins: int) -> dict:

        values = series.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=bins, random_state=42, n_init=10)
        kmeans.fit(values)
        return {"bins": bins, "centers": kmeans.cluster_centers_.flatten()}



    def _transform_equal_width(self, series: pd.Series, edge_info: dict) -> pd.Series:

        bins_count = edge_info["bins"]
        edges = edge_info["edges"]
        labels = [f"bin_{i+1}" for i in range(bins_count)]


        result = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
        return result.astype(str)

    def _transform_equal_frequency(self, series: pd.Series, edge_info: dict) -> pd.Series:

        bins_count = edge_info["bins"]
        edges = edge_info["edges"]


        actual_bins = len(edges) - 1
        labels = [f"bin_{i+1}" for i in range(actual_bins)]

        result = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
        return result.astype(str)

    def _transform_kmeans(self, series: pd.Series, edge_info: dict) -> pd.Series:

        centers = edge_info["centers"]
        values = series.values.reshape(-1, 1)


        distances = np.abs(values - centers.reshape(1, -1))
        cluster_labels = np.argmin(distances, axis=1)

        result = pd.Series([f"cluster_{l+1}" for l in cluster_labels], index=series.index)
        return result



    def _equal_width(self, series: pd.Series, bins: int) -> pd.Series:

        labels = [f"bin_{i+1}" for i in range(bins)]
        result = pd.cut(series, bins=bins, labels=labels)
        return result.astype(str)

    def _equal_frequency(self, series: pd.Series, bins: int) -> pd.Series:

        labels = [f"bin_{i+1}" for i in range(bins)]
        try:
            result = pd.qcut(series, q=bins, labels=labels, duplicates="drop")
        except ValueError:
            result = pd.qcut(series, q=bins, duplicates="drop")
            mapping = {cat: f"bin_{i+1}" for i, cat in enumerate(result.cat.categories)}
            result = result.map(mapping)
        return result.astype(str)

    def _kmeans(self, series: pd.Series, bins: int) -> pd.Series:

        values = series.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=bins, random_state=42, n_init=10)
        labels = kmeans.fit_predict(values)
        result = pd.Series([f"cluster_{l+1}" for l in labels], index=series.index)
        return result
