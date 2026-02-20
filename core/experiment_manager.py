

















from dataclasses import dataclass, field
from typing import Optional, Dict, Literal, Set, List, Union
from enum import Enum
from datetime import datetime
import logging
import random
import numpy as np
import pandas as pd

from core.models import KnowledgeBase, Fact, Rule
from core.strategies import (
    ConflictResolutionStrategy,
    FirstStrategy,
    RandomStrategy,
    SpecificityStrategy,
    RecencyStrategy
)
from core.inference import ForwardChaining, GreedyForwardChaining, BackwardChaining, InferenceResult
from core.inference_clustered import ClusteredForwardChaining
from core.clustering import RuleClusterer
from core.storage import ExperimentStorage

from preprocessing.dataset_validator import DatasetReadinessValidator
from preprocessing.discretizer import Discretizer
from preprocessing.rule_generator import RuleGenerator
from preprocessing.tree_rule_generator import TreeRuleGenerator
from preprocessing.forest_rule_generator import ForestRuleGenerator
try:
    import sys
    from pathlib import Path as PathLib

    gui_path = PathLib(__file__).parent.parent / "src"
    if str(gui_path) not in sys.path:
        sys.path.insert(0, str(gui_path))
    from firebase_service import FirebaseService
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    FirebaseService = None


logger = logging.getLogger(__name__)


class InferenceStrategy(str, Enum):

    FIRST = "First"
    RANDOM = "Random"
    SPECIFICITY = "Specificity"
    RECENCY = "Recency"


class RuleGenerationMethod(str, Enum):

    NAIVE = "Naive"
    TREE = "Tree"
    FOREST = "Forest"


class InferenceMethod(str, Enum):

    FORWARD = "Forward"
    BACKWARD = "Backward"
    GREEDY = "Greedy"


@dataclass
class ExperimentConfig:









































    seed: int
    strategy: InferenceStrategy
    generate_method: RuleGenerationMethod
    inference_method: InferenceMethod
    decision_column: str


    discretization_method: Literal["equal_width", "equal_frequency", "kmeans"] = "equal_width"
    discretization_bins: int = 3


    tree_max_depth: Optional[int] = 3
    tree_min_samples_leaf: int = 50


    forest_n_estimators: int = 10
    forest_min_depth: int = 2
    forest_max_depth: Optional[int] = 3
    forest_min_samples_leaf: int = 50


    clustering_enabled: bool = False
    n_clusters: int = 10
    centroid_method: Literal["general", "specialized", "weighted"] = "specialized"
    centroid_threshold: float = 0.3
    centroid_match_threshold: float = 0.0


    goal: Optional[Union[tuple, str]] = None


    skip_validation: bool = False

    def __post_init__(self):


        if isinstance(self.strategy, str):
            self.strategy = InferenceStrategy(self.strategy)
        if isinstance(self.generate_method, str):
            self.generate_method = RuleGenerationMethod(self.generate_method)
        if isinstance(self.inference_method, str):
            self.inference_method = InferenceMethod(self.inference_method)


        if self.seed < 0:
            raise ValueError("Seed must be non-negative")


        if self.discretization_bins <= 0:
            raise ValueError("Number of bins must be greater than 0")


        if self.generate_method == RuleGenerationMethod.TREE:
            if self.tree_max_depth is not None and self.tree_max_depth <= 0:
                raise ValueError("tree_max_depth must be greater than 0")
            if self.tree_min_samples_leaf <= 0:
                raise ValueError("tree_min_samples_leaf must be greater than 0")


        if self.generate_method == RuleGenerationMethod.FOREST:
            if self.forest_n_estimators <= 0:
                raise ValueError("forest_n_estimators must be greater than 0")

            if self.forest_min_depth <= 0:
                raise ValueError("forest_min_depth must be greater than 0")
            if self.forest_max_depth is not None and self.forest_max_depth <= 0:
                raise ValueError("forest_max_depth must be greater than 0")

            if self.forest_max_depth is not None and self.forest_min_depth > self.forest_max_depth:
                raise ValueError(f"forest_min_depth ({self.forest_min_depth}) cannot be greater than forest_max_depth ({self.forest_max_depth})")
            if self.forest_min_samples_leaf <= 0:
                raise ValueError("forest_min_samples_leaf must be greater than 0")


        if self.clustering_enabled and self.n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")


        valid_centroid_methods = ['general', 'specialized', 'weighted']
        if self.centroid_method not in valid_centroid_methods:
            raise ValueError(f"centroid_method must be one of: {valid_centroid_methods}, got: {self.centroid_method}")

        if not 0.0 <= self.centroid_threshold <= 1.0:
            raise ValueError(f"centroid_threshold must be in range [0.0, 1.0], got: {self.centroid_threshold}")

        if not 0.0 <= self.centroid_match_threshold <= 1.0:
            raise ValueError(f"centroid_match_threshold must be in range [0.0, 1.0], got: {self.centroid_match_threshold}")


        if self.inference_method == InferenceMethod.BACKWARD and self.goal is None:
            raise ValueError("Backward Chaining requires a goal to be set")

        if self.goal is not None and len(self.goal) != 2:
            raise ValueError("Goal must be a tuple (attribute, value)")

    def get_forest_params(self) -> Dict:

        return {
            "n_estimators": self.forest_n_estimators,
            "min_depth": self.forest_min_depth,
            "max_depth": self.forest_max_depth,
            "min_samples_leaf": self.forest_min_samples_leaf,
        }


class ExperimentRunner:
























    def __init__(self, config: ExperimentConfig, enable_storage: bool = True):







        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ExperimentRunner")

      
        self.enable_storage = enable_storage
        self.storage = ExperimentStorage() if enable_storage else None


        self.validator: Optional[DatasetReadinessValidator] = None
        self.discretizer: Optional[Discretizer] = None
        self.rule_generator = None
        self.clusterer: Optional[RuleClusterer] = None
        self.inference_engine = None


        self.discretized_df: Optional[pd.DataFrame] = None
        self.generated_rules: List[Rule] = []
        self.clusters: Optional[List[List[Rule]]] = None

      
        self.run_id: Optional[str] = None


        self.log_handlers: List[logging.Handler] = []

    def _setup_logging(self):









        from pathlib import Path as PathLib


        logs_dir = PathLib("logs")
        logs_dir.mkdir(exist_ok=True)


        for handler in self.log_handlers:
            self.logger.removeHandler(handler)
        self.log_handlers.clear()


        log_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


        extended_log_path = logs_dir / f"inference_{self.run_id}_extended.log"
        extended_handler = logging.FileHandler(extended_log_path, mode='w', encoding='utf-8')
        extended_handler.setLevel(logging.DEBUG)
        extended_handler.setFormatter(log_format)
        self.logger.addHandler(extended_handler)
        self.log_handlers.append(extended_handler)


        basic_log_path = logs_dir / f"inference_{self.run_id}.log"
        basic_handler = logging.FileHandler(basic_log_path, mode='w', encoding='utf-8')
        basic_handler.setLevel(logging.INFO)
        basic_handler.setFormatter(log_format)
        self.logger.addHandler(basic_handler)
        self.log_handlers.append(basic_handler)


        self.logger.setLevel(logging.DEBUG)

        self.logger.info(f"[LOGGING] Pliki logów skonfigurowane:")
        self.logger.info(f"  - Extended: {extended_log_path}")
        self.logger.info(f"  - Basic: {basic_log_path}")

    def _cleanup_logging(self):




        for handler in self.log_handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
        self.log_handlers.clear()
        self.logger.debug("[LOGGING] Handlery plików logów zamknięte")

    def run(self, df: pd.DataFrame, initial_facts: Optional[Set[Fact]] = None, dataset_name: str = "dataset") -> InferenceResult:















      
        self.run_id = self._generate_run_id()


        self._setup_logging()

        self.logger.info("="*70)
        self.logger.info("EXPERIMENT START")
        self.logger.info("="*70)
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Configuration: seed={self.config.seed}, "
                        f"strategy={self.config.strategy.value}, "
                        f"method={self.config.generate_method.value}")


        self._set_global_seed()


        if not self.config.skip_validation:
            self._validate_dataset(df)


        self._discretize_data(df)


        self._generate_rules()


        if self.config.clustering_enabled:
            self._cluster_rules()


        result = self._run_inference(initial_facts)

        self.logger.info("="*70)
        self.logger.info("EXPERIMENT COMPLETED")
        self.logger.info(f"Result: {len(result.new_facts)} new facts, "
                        f"{result.iterations} iterations, "
                        f"{result.execution_time_ms:.2f} ms")
        self.logger.info("="*70)

      
        experiment_dir = None
        if self.enable_storage and self.storage is not None:
            try:
                experiment_dir = self.storage.save_experiment(
                    run_id=self.run_id,
                    dataset_name=dataset_name,
                    config=self.config,
                    result=result,
                    rules=self.generated_rules,
                    inference_engine=self.inference_engine
                )
                self.logger.info(f"[STORAGE] Eksperyment zapisany: {experiment_dir}")
            except Exception as e:
                self.logger.error(f"[STORAGE] Błąd podczas zapisu eksperymentu: {e}", exc_info=True)


      
        if experiment_dir and FIREBASE_AVAILABLE:
            try:
                firebase = FirebaseService()


                if firebase.is_logged_in():
                    self.logger.info("[FIREBASE] Rozpoczynam synchronizację z chmurą...")


                    upload_success = firebase.upload_experiment_folder(
                        local_folder_path=experiment_dir,
                        run_id=self.run_id
                    )

                    if upload_success:
                        self.logger.info(f"[FIREBASE] ✓ Eksperyment zsynchronizowany z Firebase: {self.run_id}")
                    else:
                        self.logger.warning("[FIREBASE] Upload nie powiódł się - eksperyment zapisany tylko lokalnie")
                else:
                    self.logger.info("[FIREBASE] Użytkownik niezalogowany - pomijam synchronizację z chmurą")

            except Exception as e:
              
                self.logger.warning(f"[FIREBASE] Upload failed - saved locally only: {e}")



        self._cleanup_logging()

        return result

    def _set_global_seed(self):





        self.logger.info(f"[SEED] Ustawianie GLOBAL_SEED={self.config.seed}")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.logger.info("[SEED] GLOBAL_SEED ustawiony - reprodukowalność zapewniona")

    def _validate_dataset(self, df: pd.DataFrame):









        self.logger.info("[VALIDATION] Starting dataset validation...")
        self.validator = DatasetReadinessValidator()
        report = self.validator.validate(df, decision_column=self.config.decision_column)

        self.logger.info(f"[VALIDATION] Score: {report.score}/100, Verdict: {report.verdict}")


        critical_issues = report.get_critical_issues()
        if critical_issues:
            self.logger.error(f"[VALIDATION] Found {len(critical_issues)} critical issues:")
            for issue in critical_issues:
                self.logger.error(f"  - {issue.message}")
            raise ValueError(f"Dataset failed validation: {len(critical_issues)} critical issues")

        self.logger.info("[VALIDATION] Dataset valid")

    def _discretize_data(self, df: pd.DataFrame):






        self.logger.info(f"[DISCRETIZATION] Method: {self.config.discretization_method}, "
                        f"Bins: {self.config.discretization_bins}")

        self.discretizer = Discretizer()
        self.discretized_df = self.discretizer.fit_transform(
            df,
            method=self.config.discretization_method,
            bins=self.config.discretization_bins
        )

        self.logger.info(f"[DISCRETIZATION] Completed. Shape: {self.discretized_df.shape}")

    def _generate_rules(self):






        self.logger.info(f"[RULES] Method: {self.config.generate_method.value}")


        if self.config.decision_column not in self.discretized_df.columns:
            raise ValueError(f"Decision column '{self.config.decision_column}' does not exist in DataFrame")



        temp_generator = RuleGenerator()
        id_columns = temp_generator.detect_id_columns(self.discretized_df)


        id_columns = [col for col in id_columns if col != self.config.decision_column]

        if id_columns:
            self.logger.info(f"[RULES] Automatically detected and removed ID columns: {id_columns}")
            self.discretized_df = self.discretized_df.drop(columns=id_columns)


        if self.config.generate_method == RuleGenerationMethod.NAIVE:
            self.rule_generator = RuleGenerator()
            self.generated_rules = self.rule_generator.generate(
                self.discretized_df,
                decision_column=self.config.decision_column
            )

        elif self.config.generate_method == RuleGenerationMethod.TREE:
            self.rule_generator = TreeRuleGenerator(
                max_depth=self.config.tree_max_depth,
                min_samples_leaf=self.config.tree_min_samples_leaf
            )
            self.generated_rules = self.rule_generator.generate(
                self.discretized_df,
                decision_column=self.config.decision_column
            )

        elif self.config.generate_method == RuleGenerationMethod.FOREST:
            self.rule_generator = ForestRuleGenerator(
                n_estimators=self.config.forest_n_estimators,
                min_depth=self.config.forest_min_depth,
                max_depth=self.config.forest_max_depth,
                min_samples_leaf=self.config.forest_min_samples_leaf
            )
            self.generated_rules = self.rule_generator.generate(
                self.discretized_df,
                decision_column=self.config.decision_column
            )

        else:
            raise ValueError(f"Unknown rule generation method: {self.config.generate_method}")


        avg_premises = sum(len(r.premises) for r in self.generated_rules) / len(self.generated_rules) if self.generated_rules else 0
        self.logger.info(f"[RULES] Generated {len(self.generated_rules)} rules")
        self.logger.info(f"[RULES] Average number of premises: {avg_premises:.1f}")


        if self.generated_rules:
            self.logger.info("[RULES] Example rules:")
            for i, rule in enumerate(self.generated_rules[:3], 1):
                self.logger.info(f"  {i}. {rule}")

    def _cluster_rules(self):



        self.logger.info(f"[CLUSTERING] Number of clusters: {self.config.n_clusters}")


        n_clusters = min(self.config.n_clusters, len(self.generated_rules))

        if n_clusters < 2:
            self.logger.warning("[CLUSTERING] Too few rules for clustering, skipping...")
            self.config.clustering_enabled = False
            return

        self.clusterer = RuleClusterer(
            n_clusters=n_clusters,
            centroid_method=self.config.centroid_method,
            centroid_threshold=self.config.centroid_threshold
        )
        self.clusters = self.clusterer.fit(self.generated_rules)

        self.logger.info(f"[CLUSTERING] Created {len(self.clusters)} clusters")


        cluster_sizes = [len(cluster) for cluster in self.clusters]
        self.logger.info(f"[CLUSTERING] Cluster sizes: min={min(cluster_sizes)}, "
                        f"max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")

    def _run_inference(self, initial_facts: Optional[Set[Fact]]) -> InferenceResult:









        self.logger.info(f"[INFERENCE] Method: {self.config.inference_method.value}")


        if initial_facts is None:
            initial_facts = self._create_facts_from_row(self.discretized_df.iloc[0])
            self.logger.info(f"[INFERENCE] Created {len(initial_facts)} facts from first row")
        else:
            self.logger.info(f"[INFERENCE] Used {len(initial_facts)} initial facts")


        kb = KnowledgeBase(rules=self.generated_rules, facts=initial_facts)


        strategy = self._create_strategy()



        goal_target = None
        if self.config.goal is not None:
            if isinstance(self.config.goal, str):

                goal_target = self.config.goal
                self.logger.info(f"[INFERENCE] Goal: any value of attribute '{goal_target}'")
            else:

                goal_target = Fact(self.config.goal[0], str(self.config.goal[1]))
                self.logger.info(f"[INFERENCE] Goal: {goal_target}")


        if self.config.inference_method == InferenceMethod.BACKWARD:

            if goal_target is None:
                raise ValueError("Backward Chaining requires a goal")
            if isinstance(goal_target, str):
                raise ValueError("Backward Chaining requires a specific goal (attribute + value), "
                               "'Any Value' option is not supported for Backward Chaining")

            self.inference_engine = BackwardChaining(strategy=strategy)
            result = self.inference_engine.run(kb, goal=goal_target)

        elif self.config.inference_method == InferenceMethod.GREEDY:

            self.inference_engine = GreedyForwardChaining()
            result = self.inference_engine.run(kb, goal=goal_target)

        else:
            if self.config.clustering_enabled and self.clusters is not None:

                self.logger.info("[INFERENCE] Using ClusteredForwardChaining")
                self.inference_engine = ClusteredForwardChaining(
                    strategy=strategy,
                    clusters=self.clusters,
                    centroid_match_threshold=self.config.centroid_match_threshold
                )
                result = self.inference_engine.run(kb, goal=goal_target)
            else:

                self.inference_engine = ForwardChaining(strategy=strategy)
                result = self.inference_engine.run(kb, goal=goal_target, greedy=False)

        self.logger.info(f"[INFERENCE] Completed. Success: {result.success}, "
                        f"New facts: {len(result.new_facts)}, "
                        f"Iterations: {result.iterations}")

        return result

    def _create_strategy(self) -> ConflictResolutionStrategy:






        strategy_map = {
            InferenceStrategy.FIRST: FirstStrategy,
            InferenceStrategy.RANDOM: RandomStrategy,
            InferenceStrategy.SPECIFICITY: SpecificityStrategy,
            InferenceStrategy.RECENCY: RecencyStrategy,
        }

        strategy_class = strategy_map.get(self.config.strategy)
        if strategy_class is None:
            raise ValueError(f"Nieznana strategia: {self.config.strategy}")

        return strategy_class()

    def _create_facts_from_row(self, row: pd.Series) -> Set[Fact]:









        facts = set()

        for column, value in row.items():

            if pd.isna(value):
                continue


            value_str = str(value)


            fact = Fact(attribute=column, value=value_str)
            facts.add(fact)

        return facts

    def _generate_run_id(self) -> str:







        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"exp_{timestamp}"
        return run_id
