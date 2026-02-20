"""
Clustered Forward Chaining - Optymalizacja przez klasteryzację reguł.

Główna funkcja badawcza pracy zgodnie z A0.pdf, A8.pdf i A9.pdf.

Algorytm 2 z "Enhancing the efficiency of rule-based expert systems":
1. Oblicz podobieństwo między Faktami a Reprezentantem (Centroidem) każdego klastra
2. Wybierz klaster z MAKSYMALNYM podobieństwem (argmax)
3. Jeśli maksymalne podobieństwo = 0, zakończ. W przeciwnym razie eksploruj wybrany klaster

Zysk wydajnościowy:
- Zamiast sprawdzać 8000 reguł → sprawdzamy 50 centroidów + reguły w wybranym klastrze
- W każdej iteracji wybierany jest tylko JEDEN klaster (najbardziej pasujący)
"""

import logging
import time
from typing import Set, List, Optional

from core.models import KnowledgeBase, Fact, Rule
from core.inference import ForwardChaining, InferenceResult
from core.clustering import RuleClusterer, RuleCluster
from core.strategies import ConflictResolutionStrategy

logger = logging.getLogger(__name__)


class ClusteredForwardChaining(ForwardChaining):
    """
    Forward Chaining z optymalizacją przez klasteryzację reguł (Algorytm 2).

    Implementuje algorytm z "Enhancing the efficiency of rule-based expert systems":
    - Oblicza podobieństwo między faktami a centroidami klastrów
    - Wybiera klaster z MAKSYMALNYM podobieństwem (argmax)
    - Jeśli max similarity = 0, pomija dalsze eksplorowanie
    - Sprawdza tylko reguły w wybranym klastrze

    Dodatkowe metryki: clusters_checked, clusters_skipped

    Example:
        >>> # Najpierw sklasteryzuj reguły
        >>> clusterer = RuleClusterer(n_clusters=50)
        >>> clusters = clusterer.fit(rules)
        >>>
        >>> # Użyj w wnioskowaniu z klasteryzacją
        >>> engine = ClusteredForwardChaining(strategy, clusters)
        >>> result = engine.run(kb)
        >>> print(f"Sprawdzono {result.clusters_checked}/{len(clusters)} klastrów")
    """

    def __init__(
        self,
        strategy: ConflictResolutionStrategy,
        clusters: List[RuleCluster],
        centroid_match_threshold: float = 0.0
    ):
        """
        Inicjalizacja silnika z klasteryzacją.

        Args:
            strategy: Strategia rozwiązywania konfliktów
            clusters: Lista klastrów reguł z centroidami
            centroid_match_threshold: Minimalny próg podobieństwa (0.0-1.0)
                                     Domyślnie 0.0 - klaster wybierany gdy max > 0
                                     Zgodnie z Algorytmem 2: "jeśli max similarity = 0, stop"
        """
        super().__init__(strategy)
        self.clusters = clusters
        self.centroid_match_threshold = centroid_match_threshold

        # Dodatkowe metryki
        self.clusters_checked = 0
        self.clusters_skipped = 0
        self.centroid_evaluations = 0

    def run(self, kb: KnowledgeBase, goal: Fact = None) -> InferenceResult:
        """
        Uruchamia forward chaining z optymalizacją przez centroidy (Algorytm 2).

        Algorytm 2 z "Enhancing the efficiency...":
        1. Oblicz podobieństwo między Faktami a Centroidem każdego klastra
        2. Wybierz klaster z MAKSYMALNYM podobieństwem (argmax)
        3. Jeśli max similarity = 0, zakończ. W przeciwnym razie eksploruj wybrany klaster

        Args:
            kb: Baza wiedzy
            goal: Opcjonalny cel (dla kompatybilności, ignorowany w forward chaining)

        Returns:
            InferenceResult z dodatkowymi metrykami klasteryzacji
        """
        logger.info("=== Starting CLUSTERED Forward Chaining Inference (Algorithm 2 - argmax) ===")
        logger.info(f"Initial facts: {len(kb.facts)}, Clusters: {len(self.clusters)}, Goal: {goal}")

        start_time = time.perf_counter()

        facts = kb.facts.copy()
        new_facts: List[Fact] = []
        rules_fired: List[Rule] = []
        fired_rules_ids: Set[int] = set()  # REFRACTORINESS: Zbiór ID reguł które już odpaliły
        iterations = 0

        # Metryki wydajnościowe
        rules_evaluated = 0
        rules_activated = 0
        self.clusters_checked = 0
        self.clusters_skipped = 0
        self.centroid_evaluations = 0

        # Główna pętla wnioskowania
        while True:
            iterations += 1
            logger.debug(f"\n--- Iteration {iterations} ---")

            conflict_set = []

            # ALGORYTM 2: Oblicz podobieństwo dla WSZYSTKICH klastrów i wybierz ARGMAX
            cluster_similarities = []
            for cluster in self.clusters:
                self.centroid_evaluations += 1
                match_ratio = self._get_centroid_match_ratio(cluster.centroid, facts)
                cluster_similarities.append((cluster, match_ratio))

            # Sortuj klastry po podobieństwie malejąco
            cluster_similarities.sort(key=lambda x: x[1], reverse=True)

            # Krok 2: Wybierz klaster z MAKSYMALNYM podobieństwem
            if cluster_similarities:
                best_cluster, max_similarity = cluster_similarities[0]

                # Krok 3: Jeśli max similarity = 0, zakończ eksplorację klastrów
                if max_similarity <= self.centroid_match_threshold:
                    # Wszystkie klastry mają zerowe (lub poniżej progu) podobieństwo
                    self.clusters_skipped += len(self.clusters)
                    logger.info(f"[STOP] Max centroid similarity ({max_similarity:.2f}) <= threshold "
                              f"({self.centroid_match_threshold}) → no cluster explored")
                else:
                    # Eksploruj TYLKO najlepszy klaster (argmax)
                    self.clusters_checked += 1
                    self.clusters_skipped += len(self.clusters) - 1  # Pozostałe klastry pominięte

                    logger.debug(f"✓ ARGMAX: Cluster {best_cluster.cluster_id} selected "
                               f"(similarity={max_similarity:.2f}) → checking {best_cluster.size} rules")

                    for rule in best_cluster.rules:
                        rules_evaluated += 1
                        if (rule.id not in fired_rules_ids          # 1. Nie odpalona wcześniej (REFRACTORINESS)
                            and rule.is_satisfied_by(facts)          # 2. Przesłanki spełnione
                            and rule.conclusion not in facts):       # 3. Wnosi coś nowego
                            conflict_set.append(rule)

                    # Log pominięte klastry
                    for cluster, sim in cluster_similarities[1:]:
                        logger.debug(f"[SKIP] Cluster {cluster.cluster_id} (similarity={sim:.2f}) "
                                   f"→ {cluster.size} rules not checked")

            rules_activated += len(conflict_set)

            if not conflict_set:
                # Brak reguł do odpalenia → koniec
                logger.debug("No rules in conflict set → stopping")
                break

            logger.debug(f"Conflict Set size: {len(conflict_set)}, Rule IDs: {[r.id for r in conflict_set]}")

            # Wybierz regułę strategią
            selected_rule = self.strategy.select(conflict_set, facts)
            logger.info(f"Selected Rule {selected_rule.id}: {selected_rule}")

            # Dodaj nowy fakt
            new_fact = selected_rule.conclusion
            facts.add(new_fact)
            new_facts.append(new_fact)
            rules_fired.append(selected_rule)
            fired_rules_ids.add(selected_rule.id)  # REFRACTORINESS: Oznacz regułę jako użytą

            logger.info(f"New Fact inferred: {new_fact}")

        # Koniec wnioskowania
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        logger.info("=== Inference COMPLETED ===")
        logger.info(f"Total iterations: {iterations}, Facts: {len(facts)}, Rules fired: {len(rules_fired)}")
        logger.info(f"Execution time: {execution_time_ms:.3f} ms")
        logger.info(f"Rules evaluated: {rules_evaluated}, Rules activated: {rules_activated}")
        logger.info(f"Clusters checked: {self.clusters_checked}, Clusters skipped: {self.clusters_skipped}")
        logger.info(f"Centroid evaluations: {self.centroid_evaluations}")

        # Oblicz oszczędność (uwaga: w argmax skipped = (n_clusters - 1) * iterations)
        total_rules = sum(c.size for c in self.clusters)
        total_cluster_evaluations = self.clusters_checked + self.clusters_skipped
        if total_cluster_evaluations > 0:
            savings_percent = (self.clusters_skipped / total_cluster_evaluations * 100)
        else:
            savings_percent = 0
        logger.info(f"[Algorithm 2 - argmax] Checked {self.clusters_checked} clusters, "
                   f"skipped {self.clusters_skipped} cluster evaluations ({savings_percent:.1f}% savings)")

        return InferenceResult(
            success=len(new_facts) > 0,
            facts=facts,
            new_facts=new_facts,
            rules_fired=rules_fired,
            iterations=iterations,
            execution_time_ms=execution_time_ms,
            rules_evaluated=rules_evaluated,
            rules_activated=rules_activated,
            facts_count=len(facts)
        )

    def _get_centroid_match_ratio(self, centroid: Rule, facts: Set[Fact]) -> float:
        """
        Oblicza współczynnik dopasowania centroidu do faktów.

        Args:
            centroid: Reguła będąca centroidem
            facts: Zbiór dostępnych faktów

        Returns:
            Ratio dopasowania (0.0 - 1.0)
        """
        if len(centroid.premises) == 0:
            # Centroid bez przesłanek - zawsze pasuje
            return 1.0

        # Policz ile przesłanek centroidu jest spełnionych
        matched_premises = sum(1 for premise in centroid.premises if premise in facts)
        match_ratio = matched_premises / len(centroid.premises)

        return match_ratio

    def get_clustering_stats(self) -> dict:
        """
        Zwraca statystyki dotyczące użycia klasteryzacji (Algorytm 2 - argmax).

        W argmax: clusters_checked = liczba iteracji gdzie znaleziono pasujący klaster
                  clusters_skipped = suma pominietych klastrów we wszystkich iteracjach

        Returns:
            Słownik ze statystykami
        """
        total_rules = sum(c.size for c in self.clusters)
        total_evaluations = self.clusters_checked + self.clusters_skipped

        return {
            'total_clusters': len(self.clusters),
            'clusters_checked': self.clusters_checked,
            'clusters_skipped': self.clusters_skipped,
            'centroid_evaluations': self.centroid_evaluations,
            'skip_rate': self.clusters_skipped / total_evaluations if total_evaluations > 0 else 0.0,
            'total_rules': total_rules,
            'algorithm': 'argmax (Algorithm 2)',
        }
