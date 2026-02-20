











from typing import List, Set, Dict, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from collections import Counter

from core.models import Rule, Fact


logger = logging.getLogger(__name__)


@dataclass
class RuleCluster:









    cluster_id: int
    rules: List[Rule]
    centroid: Rule
    size: int


class RuleClusterer:














    def __init__(
        self,
        n_clusters: int = 50,
        method: str = 'agglomerative',
        linkage: str = 'average',
        random_state: int = 42,
        centroid_method: str = 'specialized',
        centroid_threshold: float = 0.3
    ):

















        self.n_clusters = n_clusters
        self.method = method
        self.linkage = linkage
        self.random_state = random_state
        self.centroid_method = centroid_method
        self.centroid_threshold = centroid_threshold
        self.clusters: List[RuleCluster] = []

    def fit(self, rules: List[Rule]) -> List[RuleCluster]:









        if len(rules) == 0:
            return []


        n_clusters = min(self.n_clusters, len(rules))

        logger.info(f"[CLUSTERING] Klasteryzacja {len(rules)} reguł na {n_clusters} klastrów (method={self.method}, centroid_method={self.centroid_method})")


        similarity_matrix = self._compute_similarity_matrix(rules)


        distance_matrix = 1.0 - similarity_matrix


        if self.method == 'agglomerative':
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage=self.linkage
            )
            labels = clusterer.fit_predict(distance_matrix)
        elif self.method == 'kmeans':

            feature_vectors = self._rules_to_feature_vectors(rules)
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            labels = clusterer.fit_predict(feature_vectors)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")


        clusters_dict: Dict[int, List[Rule]] = {}
        for rule, label in zip(rules, labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(rule)


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


        self.clusters.sort(key=lambda c: c.cluster_id)

        avg_size = sum(c.size for c in self.clusters) / len(self.clusters)
        avg_centroid_len = sum(len(c.centroid.premises) for c in self.clusters) / len(self.clusters)
        logger.info(f"[CLUSTERING] Utworzono {len(self.clusters)} klastrów, średni rozmiar: {avg_size:.1f} reguł, średnia długość centroidu: {avg_centroid_len:.1f}")

        return self.clusters

    def _compute_similarity_matrix(self, rules: List[Rule]) -> np.ndarray:
















        logger.debug(f"[CLUSTERING] Obliczanie macierzy podobieństwa (wektoryzacja)...")



        all_facts = set()
        for rule in rules:
            for premise in rule.premises:
                all_facts.add((premise.attribute, premise.value))


        fact_to_idx = {fact: idx for idx, fact in enumerate(sorted(all_facts))}
        n_facts = len(all_facts)

        logger.debug(f"[CLUSTERING] Znaleziono {n_facts} unikalnych faktów (par atrybut-wartość)")


        n_rules = len(rules)
        rule_vectors = np.zeros((n_rules, n_facts), dtype=np.int8)

        for i, rule in enumerate(rules):
            for premise in rule.premises:
                fact = (premise.attribute, premise.value)
                if fact in fact_to_idx:
                    j = fact_to_idx[fact]
                    rule_vectors[i, j] = 1


        from sklearn.metrics import pairwise_distances


        distance_matrix = pairwise_distances(rule_vectors, metric='jaccard', n_jobs=-1)
        similarity_matrix = 1.0 - distance_matrix

        logger.debug(f"[CLUSTERING] Macierz podobieństwa {n_rules}x{n_rules} obliczona")

        return similarity_matrix

    def _rules_to_feature_vectors(self, rules: List[Rule]) -> np.ndarray:
















        all_facts = set()
        for rule in rules:
            for premise in rule.premises:
                all_facts.add((premise.attribute, premise.value))

        fact_to_idx = {fact: idx for idx, fact in enumerate(sorted(all_facts))}


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









        if not cluster_rules:
            raise ValueError("Cannot compute centroid for empty cluster")


        conclusion_counter = Counter(
            (rule.conclusion.attribute, rule.conclusion.value)
            for rule in cluster_rules
        )
        most_common_conclusion = conclusion_counter.most_common(1)[0][0]
        conclusion = Fact(most_common_conclusion[0], most_common_conclusion[1])



        common_premises = set(
            (premise.attribute, premise.value)
            for premise in cluster_rules[0].premises
        )


        for rule in cluster_rules[1:]:
            rule_premises = set(
                (premise.attribute, premise.value)
                for premise in rule.premises
            )
            common_premises &= rule_premises


        centroid_premises = [
            Fact(attr, val) for (attr, val) in sorted(common_premises)
        ]



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








        if not cluster_rules:
            raise ValueError("Cannot compute centroid for empty cluster")


        conclusion_counter = Counter(
            (rule.conclusion.attribute, rule.conclusion.value)
            for rule in cluster_rules
        )
        most_common_conclusion = conclusion_counter.most_common(1)[0][0]
        conclusion = Fact(most_common_conclusion[0], most_common_conclusion[1])


        all_premises = set()
        for rule in cluster_rules:
            for premise in rule.premises:
                all_premises.add((premise.attribute, premise.value))


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









        if not cluster_rules:
            raise ValueError("Cannot compute centroid for empty cluster")


        conclusion_counter = Counter(
            (rule.conclusion.attribute, rule.conclusion.value)
            for rule in cluster_rules
        )
        most_common_conclusion = conclusion_counter.most_common(1)[0][0]
        conclusion = Fact(most_common_conclusion[0], most_common_conclusion[1])


        fact_counter = Counter()
        for rule in cluster_rules:
            for premise in rule.premises:
                fact_counter[(premise.attribute, premise.value)] += 1


        n_rules = len(cluster_rules)



        centroid_premises = []
        for (attribute, value), count in fact_counter.items():
            frequency = count / n_rules
            if frequency >= threshold:
                centroid_premises.append(Fact(attribute, value))


        centroid_premises.sort(key=lambda f: (f.attribute, f.value))



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












        for cluster in self.clusters:
            if rule in cluster.rules:
                return cluster
        raise ValueError(f"Rule {rule.id} not found in any cluster")

    def get_statistics(self) -> Dict:






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
