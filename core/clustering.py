"""
Rule Clustering Module - Główna funkcja badawcza pracy.

Implementuje algorytm klasteryzacji reguł dla optymalizacji forward chaining:
1. Grupuje podobne reguły w klastry
2. Wyznacza reprezentantów (centroidy) dla każdego klastra
3. Umożliwia szybkie wnioskowanie przez sprawdzanie centroidów

Zgodnie z A0.pdf i A9.pdf: Centroidy jako "filtry" redukujące liczbę
sprawdzanych reguł podczas wnioskowania.
"""

from typing import List, Set, Dict, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from collections import Counter

from core.models import Rule, Fact

# Logger dla modułu klasteryzacji
logger = logging.getLogger(__name__)


@dataclass
class RuleCluster:
    """
    Klaster reguł z wyznaczonym centroidem.

    Attributes:
        cluster_id: Identyfikator klastra
        rules: Lista reguł należących do klastra
        centroid: Reprezentant klastra (sztuczna reguła - część wspólna)
        size: Liczba reguł w klastrze
    """
    cluster_id: int
    rules: List[Rule]
    centroid: Rule
    size: int


class RuleClusterer:
    """
    Klasteryzacja reguł dla optymalizacji wnioskowania.

    Algorytm:
    1. Oblicza macierz podobieństwa między regułami (Jaccard Index)
    2. Grupuje reguły w klastry (AgglomerativeClustering lub KMeans)
    3. Dla każdego klastra wyznacza centroid (część wspólna przesłanek)

    Example:
        >>> clusterer = RuleClusterer(n_clusters=50, method='agglomerative')
        >>> clusters = clusterer.fit(rules)
        >>> # Teraz można używać clusters w ClusteredForwardChaining
    """

    def __init__(
        self,
        n_clusters: int = 50,
        method: str = 'agglomerative',
        linkage: str = 'average',
        random_state: int = 42,
        centroid_method: str = 'specialized',
        centroid_threshold: float = 0.3
    ):
        """
        Inicjalizacja klasteryzatora reguł.

        Args:
            n_clusters: Docelowa liczba klastrów (np. 50 dla 8000 reguł = ~160 reguł/klaster)
            method: Metoda klasteryzacji ('agglomerative' lub 'kmeans')
            linkage: Typ linkage dla agglomerative ('average', 'complete', 'single')
            random_state: Seed dla powtarzalności
            centroid_method: Metoda obliczania centroidów zgodnie z A8.pdf:
                - 'general': Dolne przybliżenie - tylko wspólne przesłanki (∩)
                             UWAGA: Może powodować "Empty Representative" problem!
                - 'specialized': Górne przybliżenie - wszystkie przesłanki (∪)
                                 ZALECANE: Nigdy nie daje pustego centroidu
                - 'weighted': Threshold-based - przesłanki z progiem częstości
                              Użyj niskiego progu (np. 0.3) aby uniknąć pustych centroidów
            centroid_threshold: Próg częstości dla metody 'weighted' (domyślnie 0.3 = 30%)
        """
        self.n_clusters = n_clusters
        self.method = method
        self.linkage = linkage
        self.random_state = random_state
        self.centroid_method = centroid_method
        self.centroid_threshold = centroid_threshold
        self.clusters: List[RuleCluster] = []

    def fit(self, rules: List[Rule]) -> List[RuleCluster]:
        """
        Klasteryzuje reguły i wyznacza centroidy.

        Args:
            rules: Lista wszystkich reguł do sklasteryzowania

        Returns:
            Lista klastrów z centroidami
        """
        if len(rules) == 0:
            return []

        # Dostosuj liczbę klastrów jeśli mamy mniej reguł
        n_clusters = min(self.n_clusters, len(rules))

        logger.info(f"[CLUSTERING] Klasteryzacja {len(rules)} reguł na {n_clusters} klastrów (method={self.method}, centroid_method={self.centroid_method})")

        # 1. Oblicz macierz podobieństwa
        similarity_matrix = self._compute_similarity_matrix(rules)

        # 2. Konwertuj podobieństwo na odległość (distance = 1 - similarity)
        distance_matrix = 1.0 - similarity_matrix

        # 3. Wykonaj klasteryzację
        if self.method == 'agglomerative':
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage=self.linkage
            )
            labels = clusterer.fit_predict(distance_matrix)
        elif self.method == 'kmeans':
            # KMeans działa na wektorach cech, nie macierzy odległości
            feature_vectors = self._rules_to_feature_vectors(rules)
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            labels = clusterer.fit_predict(feature_vectors)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # 4. Grupuj reguły w klastry
        clusters_dict: Dict[int, List[Rule]] = {}
        for rule, label in zip(rules, labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(rule)

        # 5. Wyznacz centroidy dla każdego klastra
        self.clusters = []
        for cluster_id, cluster_rules in clusters_dict.items():
            centroid = self._compute_centroid(cluster_rules, cluster_id)
            cluster = RuleCluster(
                cluster_id=cluster_id,
                rules=cluster_rules,
                centroid=centroid,
                size=len(cluster_rules)
            )
            self.clusters.append(cluster)

        # Sortuj klastry po ID dla spójności
        self.clusters.sort(key=lambda c: c.cluster_id)

        avg_size = sum(c.size for c in self.clusters) / len(self.clusters)
        avg_centroid_len = sum(len(c.centroid.premises) for c in self.clusters) / len(self.clusters)
        logger.info(f"[CLUSTERING] Utworzono {len(self.clusters)} klastrów, średni rozmiar: {avg_size:.1f} reguł, średnia długość centroidu: {avg_centroid_len:.1f}")

        return self.clusters

    def _compute_similarity_matrix(self, rules: List[Rule]) -> np.ndarray:
        """
        Oblicza macierz podobieństwa między regułami (Jaccard Index).

        POPRAWIONE: Używa par (atrybut, wartość) zamiast samych atrybutów!

        Jaccard Index dla reguł:
        similarity(R1, R2) = |facts(R1) ∩ facts(R2)| / |facts(R1) ∪ facts(R2)|

        gdzie facts(R) = zbiór par (attribute, value) z przesłanek reguły R

        Args:
            rules: Lista reguł

        Returns:
            Macierz podobieństwa NxN gdzie N = len(rules)
        """
        logger.debug(f"[CLUSTERING] Obliczanie macierzy podobieństwa (wektoryzacja)...")

        # OPTYMALIZACJA: Użyj wektoryzacji zamiast pętli O(N^2)
        # Krok 1: Zbierz wszystkie unikalne fakty (pary attr-value)
        all_facts = set()
        for rule in rules:
            for premise in rule.premises:
                all_facts.add((premise.attribute, premise.value))

        # Krok 2: Stwórz mapowanie fakt → indeks
        fact_to_idx = {fact: idx for idx, fact in enumerate(sorted(all_facts))}
        n_facts = len(all_facts)

        logger.debug(f"[CLUSTERING] Znaleziono {n_facts} unikalnych faktów (par atrybut-wartość)")

        # Krok 3: Konwertuj każdą regułę na wektor binarny
        n_rules = len(rules)
        rule_vectors = np.zeros((n_rules, n_facts), dtype=np.int8)

        for i, rule in enumerate(rules):
            for premise in rule.premises:
                fact = (premise.attribute, premise.value)
                if fact in fact_to_idx:
                    j = fact_to_idx[fact]
                    rule_vectors[i, j] = 1

        # Krok 4: Oblicz Jaccard używając sklearn (szybkie!)
        from sklearn.metrics import pairwise_distances

        # pairwise_distances zwraca ODLEGŁOŚĆ, więc similarity = 1 - distance
        distance_matrix = pairwise_distances(rule_vectors, metric='jaccard', n_jobs=-1)
        similarity_matrix = 1.0 - distance_matrix

        logger.debug(f"[CLUSTERING] Macierz podobieństwa {n_rules}x{n_rules} obliczona")

        return similarity_matrix

    def _rules_to_feature_vectors(self, rules: List[Rule]) -> np.ndarray:
        """
        Konwertuje reguły na wektory cech dla KMeans.

        POPRAWIONE: Używa par (atrybut, wartość) zamiast samych atrybutów!

        Wektor cech dla reguły:
        - Każda para (atrybut, wartość) = 1 cecha (0/1)
        - Wartość 1 jeśli para występuje w przesłankach

        Args:
            rules: Lista reguł

        Returns:
            Macierz cech NxM gdzie N=liczba reguł, M=liczba unikalnych par (attr, val)
        """
        # Zbierz wszystkie unikalne fakty (pary attr-value)
        all_facts = set()
        for rule in rules:
            for premise in rule.premises:
                all_facts.add((premise.attribute, premise.value))

        fact_to_idx = {fact: idx for idx, fact in enumerate(sorted(all_facts))}

        # Utwórz macierz binarną
        n_rules = len(rules)
        n_facts = len(all_facts)
        features = np.zeros((n_rules, n_facts), dtype=np.int8)

        for i, rule in enumerate(rules):
            for premise in rule.premises:
                fact = (premise.attribute, premise.value)
                if fact in fact_to_idx:
                    j = fact_to_idx[fact]
                    features[i, j] = 1

        return features

    def _compute_centroid(self, cluster_rules: List[Rule], cluster_id: int) -> Rule:
        """
        Wyznacza centroid klastra używając wybranej metody.

        Metody zgodnie z A8.pdf (str. 103-104):
        - 'general': Dolne przybliżenie (tylko wspólne przesłanki)
        - 'specialized': Górne przybliżenie (wszystkie przesłanki)
        - 'weighted': Threshold-based (przesłanki z wagami)

        Args:
            cluster_rules: Reguły w klastrze
            cluster_id: ID klastra (dla ID centroidu)

        Returns:
            Reguła będąca centroidem klastra
        """
        if self.centroid_method == 'general':
            return self._compute_centroid_general(cluster_rules, cluster_id)
        elif self.centroid_method == 'specialized':
            return self._compute_centroid_specialized(cluster_rules, cluster_id)
        elif self.centroid_method == 'weighted':
            return self._compute_centroid_weighted(cluster_rules, cluster_id, self.centroid_threshold)
        else:
            raise ValueError(f"Unknown centroid method: {self.centroid_method}")

    def _compute_centroid_general(
        self,
        cluster_rules: List[Rule],
        cluster_id: int
    ) -> Rule:
        """
        Profile_general: TYLKO wspólne przesłanki (∩).

        Zgodnie z A8.pdf (str. 103):
        Profile_general(Rj) = ∩{ps : ∀ri∈Rj ps ∈ cond(ri)}

        Zawiera TYLKO te przesłanki, które występują we WSZYSTKICH regułach klastra.
        Jeśli brak wspólnych przesłanek, używa najbardziej popularnej przesłanki (fallback).
        """
        if not cluster_rules:
            raise ValueError("Cannot compute centroid for empty cluster")

        # 1. Znajdź najbardziej popularną konkluzję
        conclusion_counter = Counter(
            (rule.conclusion.attribute, rule.conclusion.value)
            for rule in cluster_rules
        )
        most_common_conclusion = conclusion_counter.most_common(1)[0][0]
        conclusion = Fact(most_common_conclusion[0], most_common_conclusion[1])

        # 2. Znajdź przesłanki które występują we WSZYSTKICH regułach
        # Zaczynamy od przesłanek pierwszej reguły
        common_premises = set(
            (premise.attribute, premise.value)
            for premise in cluster_rules[0].premises
        )

        # Robimy przecięcie z przesłankami każdej kolejnej reguły
        for rule in cluster_rules[1:]:
            rule_premises = set(
                (premise.attribute, premise.value)
                for premise in rule.premises
            )
            common_premises &= rule_premises  # Przecięcie (∩)

        # 3. Utwórz centroid z wspólnych przesłanek
        centroid_premises = [
            Fact(attr, val) for (attr, val) in sorted(common_premises)
        ]

        # FALLBACK: Jeśli brak wspólnych przesłanek, użyj najbardziej popularnej
        # (Rule wymaga niepustych premises)
        if not centroid_premises:
            fact_counter = Counter()
            for rule in cluster_rules:
                for premise in rule.premises:
                    fact_counter[(premise.attribute, premise.value)] += 1
            most_common = fact_counter.most_common(1)[0][0]
            centroid_premises = [Fact(most_common[0], most_common[1])]

        centroid_id = 1_000_000 + cluster_id

        return Rule(
            id=centroid_id,
            premises=centroid_premises,
            conclusion=conclusion
        )

    def _compute_centroid_specialized(
        self,
        cluster_rules: List[Rule],
        cluster_id: int
    ) -> Rule:
        """
        Profile_specialized: WSZYSTKIE przesłanki (∪).

        Zgodnie z A8.pdf (str. 103):
        Profile_specialized(Rj) = ∪{ps : ∃ri∈Rj ps ∈ cond(ri)}

        Zawiera WSZYSTKIE przesłanki, które wystąpiły PRZYNAJMNIEJ RAZ w klastrze.
        """
        if not cluster_rules:
            raise ValueError("Cannot compute centroid for empty cluster")

        # 1. Znajdź najbardziej popularną konkluzję
        conclusion_counter = Counter(
            (rule.conclusion.attribute, rule.conclusion.value)
            for rule in cluster_rules
        )
        most_common_conclusion = conclusion_counter.most_common(1)[0][0]
        conclusion = Fact(most_common_conclusion[0], most_common_conclusion[1])

        # 2. Zbierz WSZYSTKIE unikalne przesłanki z całego klastra
        all_premises = set()
        for rule in cluster_rules:
            for premise in rule.premises:
                all_premises.add((premise.attribute, premise.value))

        # 3. Utwórz centroid ze wszystkich przesłanek
        centroid_premises = [
            Fact(attr, val) for (attr, val) in sorted(all_premises)
        ]

        centroid_id = 1_000_000 + cluster_id

        return Rule(
            id=centroid_id,
            premises=centroid_premises,
            conclusion=conclusion
        )

    def _compute_centroid_weighted(
        self,
        cluster_rules: List[Rule],
        cluster_id: int,
        threshold: float = 0.5
    ) -> Rule:
        """
        Profile_weighted: Przesłanki z progiem częstości.

        Zgodnie z A8.pdf (str. 104):
        Profile_weighted(Rj) = ∪{ps : ∀ri∈Rj (npsRj / nRj) ≥ T}

        Zawiera przesłanki które występują w ≥ T% reguł klastra.
        Przykład: threshold=0.5 → przesłanka musi być w ≥50% reguł.
        """
        if not cluster_rules:
            raise ValueError("Cannot compute centroid for empty cluster")

        # 1. Znajdź najbardziej popularną konkluzję
        conclusion_counter = Counter(
            (rule.conclusion.attribute, rule.conclusion.value)
            for rule in cluster_rules
        )
        most_common_conclusion = conclusion_counter.most_common(1)[0][0]
        conclusion = Fact(most_common_conclusion[0], most_common_conclusion[1])

        # 2. Policz częstość wystąpienia każdej pary (atrybut, wartość)
        fact_counter = Counter()
        for rule in cluster_rules:
            for premise in rule.premises:
                fact_counter[(premise.attribute, premise.value)] += 1

        # 3. Wybierz tylko te przesłanki które przekraczają próg
        n_rules = len(cluster_rules)
        # POPRAWIONE: używamy float porównania zamiast int (count/n_rules >= threshold)
        # to gwarantuje poprawne zachowanie np. dla 2/3=0.67 >= 0.5

        centroid_premises = []
        for (attribute, value), count in fact_counter.items():
            frequency = count / n_rules
            if frequency >= threshold:
                centroid_premises.append(Fact(attribute, value))

        # Sortuj dla deterministyczności
        centroid_premises.sort(key=lambda f: (f.attribute, f.value))

        # FALLBACK: Jeśli żadna przesłanka nie przekracza progu, użyj najbardziej popularnej
        # (Rule wymaga niepustych premises)
        if not centroid_premises:
            most_common = fact_counter.most_common(1)[0][0]
            centroid_premises = [Fact(most_common[0], most_common[1])]

        centroid_id = 1_000_000 + cluster_id

        return Rule(
            id=centroid_id,
            premises=centroid_premises,
            conclusion=conclusion
        )

    def get_cluster_for_rule(self, rule: Rule) -> RuleCluster:
        """
        Znajduje klaster do którego należy dana reguła.

        Args:
            rule: Reguła do wyszukania

        Returns:
            RuleCluster zawierający tę regułę

        Raises:
            ValueError: Jeśli reguła nie należy do żadnego klastra
        """
        for cluster in self.clusters:
            if rule in cluster.rules:
                return cluster
        raise ValueError(f"Rule {rule.id} not found in any cluster")

    def get_statistics(self) -> Dict:
        """
        Zwraca statystyki klasteryzacji.

        Returns:
            Słownik ze statystykami
        """
        if not self.clusters:
            return {}

        sizes = [c.size for c in self.clusters]
        centroid_lengths = [len(c.centroid.premises) for c in self.clusters]

        return {
            'n_clusters': len(self.clusters),
            'total_rules': sum(sizes),
            'avg_cluster_size': np.mean(sizes),
            'min_cluster_size': min(sizes),
            'max_cluster_size': max(sizes),
            'std_cluster_size': np.std(sizes),
            'avg_centroid_length': np.mean(centroid_lengths),
            'min_centroid_length': min(centroid_lengths),
            'max_centroid_length': max(centroid_lengths),
            'centroid_method': self.centroid_method,
            'centroid_threshold': self.centroid_threshold if self.centroid_method == 'weighted' else None,
        }
