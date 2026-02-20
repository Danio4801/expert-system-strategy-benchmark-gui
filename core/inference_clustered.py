














import logging
import time
from typing import Set, List, Optional

from core.models import KnowledgeBase, Fact, Rule
from core.inference import ForwardChaining, InferenceResult
from core.clustering import RuleClusterer, RuleCluster
from core.strategies import ConflictResolutionStrategy

logger = logging.getLogger(__name__)


class ClusteredForwardChaining(ForwardChaining):






















    def __init__(
        self,
        strategy: ConflictResolutionStrategy,
        clusters: List[RuleCluster],
        centroid_match_threshold: float = 0.0
    ):










        super().__init__(strategy)
        self.clusters = clusters
        self.centroid_match_threshold = centroid_match_threshold


        self.clusters_checked = 0
        self.clusters_skipped = 0
        self.centroid_evaluations = 0

    def run(self, kb: KnowledgeBase, goal: Fact = None) -> InferenceResult:















        logger.info("=== Starting CLUSTERED Forward Chaining Inference (Algorithm 2 - argmax) ===")
        logger.info(f"Initial facts: {len(kb.facts)}, Clusters: {len(self.clusters)}, Goal: {goal}")

        start_time = time.perf_counter()

        facts = kb.facts.copy()
        new_facts: List[Fact] = []
        rules_fired: List[Rule] = []
        fired_rules_ids: Set[int] = set()
        iterations = 0


        rules_evaluated = 0
        rules_activated = 0
        self.clusters_checked = 0
        self.clusters_skipped = 0
        self.centroid_evaluations = 0


        while True:
            iterations += 1
            logger.debug(f"\n--- Iteration {iterations} ---")

            conflict_set = []


            cluster_similarities = []
            for cluster in self.clusters:
                self.centroid_evaluations += 1
                match_ratio = self._get_centroid_match_ratio(cluster.centroid, facts)
                cluster_similarities.append((cluster, match_ratio))


            cluster_similarities.sort(key=lambda x: x[1], reverse=True)


            if cluster_similarities:
                best_cluster, max_similarity = cluster_similarities[0]


                if max_similarity <= self.centroid_match_threshold:

                    self.clusters_skipped += len(self.clusters)
                    logger.info(f"[STOP] Max centroid similarity ({max_similarity:.2f}) <= threshold "
                              f"({self.centroid_match_threshold}) → no cluster explored")
                else:

                    self.clusters_checked += 1
                    self.clusters_skipped += len(self.clusters) - 1

                    logger.debug(f"✓ ARGMAX: Cluster {best_cluster.cluster_id} selected "
                               f"(similarity={max_similarity:.2f}) → checking {best_cluster.size} rules")

                    for rule in best_cluster.rules:
                        rules_evaluated += 1
                        if (rule.id not in fired_rules_ids
                            and rule.is_satisfied_by(facts)
                            and rule.conclusion not in facts):
                            conflict_set.append(rule)


                    for cluster, sim in cluster_similarities[1:]:
                        logger.debug(f"[SKIP] Cluster {cluster.cluster_id} (similarity={sim:.2f}) "
                                   f"→ {cluster.size} rules not checked")

            rules_activated += len(conflict_set)

            if not conflict_set:

                logger.debug("No rules in conflict set → stopping")
                break

            logger.debug(f"Conflict Set size: {len(conflict_set)}, Rule IDs: {[r.id for r in conflict_set]}")


            selected_rule = self.strategy.select(conflict_set, facts)
            logger.info(f"Selected Rule {selected_rule.id}: {selected_rule}")


            new_fact = selected_rule.conclusion
            facts.add(new_fact)
            new_facts.append(new_fact)
            rules_fired.append(selected_rule)
            fired_rules_ids.add(selected_rule.id)

            logger.info(f"New Fact inferred: {new_fact}")


        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        logger.info("=== Inference COMPLETED ===")
        logger.info(f"Total iterations: {iterations}, Facts: {len(facts)}, Rules fired: {len(rules_fired)}")
        logger.info(f"Execution time: {execution_time_ms:.3f} ms")
        logger.info(f"Rules evaluated: {rules_evaluated}, Rules activated: {rules_activated}")
        logger.info(f"Clusters checked: {self.clusters_checked}, Clusters skipped: {self.clusters_skipped}")
        logger.info(f"Centroid evaluations: {self.centroid_evaluations}")


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










        if len(centroid.premises) == 0:

            return 1.0


        matched_premises = sum(1 for premise in centroid.premises if premise in facts)
        match_ratio = matched_premises / len(centroid.premises)

        return match_ratio

    def get_clustering_stats(self) -> dict:









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
