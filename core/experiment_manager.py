"""
Experiment Manager - Orkiestracja eksperymentów wnioskowania.

Moduł zawiera:
    - ExperimentConfig: Konfiguracja eksperymentu (dataclass)
    - ExperimentRunner: Wykonuje pełny pipeline eksperymentu

Pipeline eksperymentu:
    1. Walidacja danych (DatasetValidator)
    2. Dyskretyzacja (Discretizer)
    3. Generowanie reguł (RuleGenerator/TreeRuleGenerator/ForestRuleGenerator)
    4. Klasteryzacja reguł - opcjonalnie (RuleClusterer)
    5. Wnioskowanie (ForwardChaining/BackwardChaining/GreedyForwardChaining/ClusteredForwardChaining)

KRYTYCZNE: Moduł musi przestrzegać zasad reprodukowalności - GLOBAL_SEED
ustawiany przed wszystkimi operacjami losowymi.
"""

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
    # Add src to path if not already there
    gui_path = PathLib(__file__).parent.parent / "src"
    if str(gui_path) not in sys.path:
        sys.path.insert(0, str(gui_path))
    from firebase_service import FirebaseService
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    FirebaseService = None

# Logger dla modułu
logger = logging.getLogger(__name__)


class InferenceStrategy(str, Enum):
    """Enum dla strategii conflict resolution."""
    FIRST = "First"
    RANDOM = "Random"
    SPECIFICITY = "Specificity"
    RECENCY = "Recency"


class RuleGenerationMethod(str, Enum):
    """Enum dla metod generowania reguł."""
    NAIVE = "Naive"
    TREE = "Tree"
    FOREST = "Forest"


class InferenceMethod(str, Enum):
    """Enum dla metod wnioskowania."""
    FORWARD = "Forward"
    BACKWARD = "Backward"
    GREEDY = "Greedy"


@dataclass
class ExperimentConfig:
    """
    Konfiguracja eksperymentu wnioskowania.

    Attributes:
        seed: Seed dla reprodukowalności (KRYTYCZNE - musi być ustawione przed losowaniem)
        strategy: Strategia conflict resolution (First, Random, Specificity, Recency)
        generate_method: Metoda generowania reguł (Naive, Tree, Forest)
        inference_method: Metoda wnioskowania (Forward, Backward, Greedy)
        decision_column: Nazwa kolumny decyzyjnej w datasecie

        # Parametry dyskretyzacji
        discretization_method: Metoda dyskretyzacji ("equal_width", "equal_frequency", "kmeans")
        discretization_bins: Liczba binów dla dyskretyzacji

        # Parametry generowania reguł (Tree)
        tree_max_depth: Maksymalna głębokość drzewa (dla Tree)
        tree_min_samples_leaf: Min. próbek w liściu (dla Tree)

        # Parametry generowania reguł (Forest)
        forest_n_estimators: Liczba drzew w lesie (dla Forest)
        forest_min_depth: Minimalna głębokość drzewa (dla Forest - Variable Depth)
        forest_max_depth: Maksymalna głębokość drzewa (dla Forest)
        forest_min_samples_leaf: Min. próbek w liściu (dla Forest)

        # Klasteryzacja
        clustering_enabled: Czy włączyć klasteryzację reguł
        n_clusters: Liczba klastrów (jeśli clustering_enabled=True)
        centroid_method: Metoda obliczania centroidów ('general', 'specialized', 'weighted')
        centroid_threshold: Próg częstości dla metody 'weighted' (0.0-1.0)
        centroid_match_threshold: Próg dopasowania centroidu podczas wnioskowania (0.0-1.0)

        # Cel wnioskowania (dla Backward lub Forward z celem)
        goal: Opcjonalny cel wnioskowania:
              - tuple (atrybut, wartość): zatrzymaj gdy znajdzie konkretny fakt
              - str (nazwa atrybutu): zatrzymaj gdy znajdzie DOWOLNĄ wartość tego atrybutu

        # Walidacja datasetu
        skip_validation: Czy pominąć walidację datasetu (domyślnie False)
    """

    # WYMAGANE
    seed: int
    strategy: InferenceStrategy
    generate_method: RuleGenerationMethod
    inference_method: InferenceMethod
    decision_column: str

    # Dyskretyzacja
    discretization_method: Literal["equal_width", "equal_frequency", "kmeans"] = "equal_width"
    discretization_bins: int = 3

    # Tree parameters
    tree_max_depth: Optional[int] = 3
    tree_min_samples_leaf: int = 50

    # Forest parameters
    forest_n_estimators: int = 10
    forest_min_depth: int = 2
    forest_max_depth: Optional[int] = 3
    forest_min_samples_leaf: int = 50

    # Klasteryzacja
    clustering_enabled: bool = False
    n_clusters: int = 10
    centroid_method: Literal["general", "specialized", "weighted"] = "specialized"  # UpperApp - unika "Empty Representative"
    centroid_threshold: float = 0.3  # Niski próg dla 'weighted' (unika pustych centroidów)
    centroid_match_threshold: float = 0.0  # Algorytm 2 argmax: 0.0 = akceptuj każdy klaster z similarity > 0

    # Cel wnioskowania (tuple lub string dla "Any Value")
    goal: Optional[Union[tuple, str]] = None

    # Walidacja
    skip_validation: bool = False

    def __post_init__(self):
        """Walidacja konfiguracji."""
        # Konwersja stringów na enumy jeśli przekazano stringi
        if isinstance(self.strategy, str):
            self.strategy = InferenceStrategy(self.strategy)
        if isinstance(self.generate_method, str):
            self.generate_method = RuleGenerationMethod(self.generate_method)
        if isinstance(self.inference_method, str):
            self.inference_method = InferenceMethod(self.inference_method)

        # Walidacja seed
        if self.seed < 0:
            raise ValueError("Seed must be non-negative")

        # Walidacja bins
        if self.discretization_bins <= 0:
            raise ValueError("Number of bins must be greater than 0")

        # Walidacja parametrów Tree
        if self.generate_method == RuleGenerationMethod.TREE:
            if self.tree_max_depth is not None and self.tree_max_depth <= 0:
                raise ValueError("tree_max_depth must be greater than 0")
            if self.tree_min_samples_leaf <= 0:
                raise ValueError("tree_min_samples_leaf must be greater than 0")

        # Walidacja parametrów Forest
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

        # Walidacja klasteryzacji
        if self.clustering_enabled and self.n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")

        # Walidacja parametrów centroidu (zgodnie z A8.pdf)
        valid_centroid_methods = ['general', 'specialized', 'weighted']
        if self.centroid_method not in valid_centroid_methods:
            raise ValueError(f"centroid_method must be one of: {valid_centroid_methods}, got: {self.centroid_method}")

        if not 0.0 <= self.centroid_threshold <= 1.0:
            raise ValueError(f"centroid_threshold must be in range [0.0, 1.0], got: {self.centroid_threshold}")

        if not 0.0 <= self.centroid_match_threshold <= 1.0:
            raise ValueError(f"centroid_match_threshold must be in range [0.0, 1.0], got: {self.centroid_match_threshold}")

        # Walidacja celu
        if self.inference_method == InferenceMethod.BACKWARD and self.goal is None:
            raise ValueError("Backward Chaining requires a goal to be set")

        if self.goal is not None and len(self.goal) != 2:
            raise ValueError("Goal must be a tuple (attribute, value)")

    def get_forest_params(self) -> Dict:
        """Zwraca parametry dla ForestRuleGenerator."""
        return {
            "n_estimators": self.forest_n_estimators,
            "min_depth": self.forest_min_depth,
            "max_depth": self.forest_max_depth,
            "min_samples_leaf": self.forest_min_samples_leaf,
        }


class ExperimentRunner:
    """
    Runner eksperymentu wnioskowania.

    Wykonuje pełny pipeline:
        1. Ustawienie GLOBAL_SEED (KRYTYCZNE dla reprodukowalności)
        2. Walidacja datasetu (opcjonalnie)
        3. Dyskretyzacja danych
        4. Generowanie reguł (Naive/Tree/Forest)
        5. Klasteryzacja reguł (opcjonalnie)
        6. Wnioskowanie (Forward/Backward/Greedy/Clustered)

    Example:
        >>> config = ExperimentConfig(
        ...     seed=42,
        ...     strategy=InferenceStrategy.FIRST,
        ...     generate_method=RuleGenerationMethod.TREE,
        ...     inference_method=InferenceMethod.FORWARD,
        ...     decision_column="class"
        ... )
        >>> runner = ExperimentRunner(config)
        >>> result = runner.run(df)
        >>> print(f"Nowych faktów: {len(result.new_facts)}")
    """

    def __init__(self, config: ExperimentConfig, enable_storage: bool = True):
        """
        Inicjalizuje runner.

        Args:
            config: Konfiguracja eksperymentu
            enable_storage: Czy automatycznie zapisywać eksperymenty na dysk (domyślnie True)
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ExperimentRunner")

      
        self.enable_storage = enable_storage
        self.storage = ExperimentStorage() if enable_storage else None

        # Komponenty pipeline (będą zainicjalizowane w run())
        self.validator: Optional[DatasetReadinessValidator] = None
        self.discretizer: Optional[Discretizer] = None
        self.rule_generator = None  # RuleGenerator | TreeRuleGenerator | ForestRuleGenerator
        self.clusterer: Optional[RuleClusterer] = None
        self.inference_engine = None  # ForwardChaining | BackwardChaining | etc.

        # Dane pośrednie
        self.discretized_df: Optional[pd.DataFrame] = None
        self.generated_rules: List[Rule] = []
        self.clusters: Optional[List[List[Rule]]] = None

      
        self.run_id: Optional[str] = None

        # Handlery plików logów (będą ustawione w _setup_logging)
        self.log_handlers: List[logging.Handler] = []

    def _setup_logging(self):
        """
        Konfiguruje logowanie do plików dla tego eksperymentu.

        Tworzy dwa pliki logów w katalogu logs/:
        - inference_{run_id}_extended.log (DEBUG level - szczegółowe logi XAI)
        - inference_{run_id}.log (INFO level - podstawowe logi)

        UWAGA: Metoda musi być wywołana PO wygenerowaniu run_id!
        """
        from pathlib import Path as PathLib

        # Utwórz katalog logs/ jeśli nie istnieje
        logs_dir = PathLib("logs")
        logs_dir.mkdir(exist_ok=True)

        # Usuń poprzednie handlery z tego runnera (jeśli istnieją)
        for handler in self.log_handlers:
            self.logger.removeHandler(handler)
        self.log_handlers.clear()

        # Format logów
        log_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Handler 1: Extended log (DEBUG level - wszystko)
        extended_log_path = logs_dir / f"inference_{self.run_id}_extended.log"
        extended_handler = logging.FileHandler(extended_log_path, mode='w', encoding='utf-8')
        extended_handler.setLevel(logging.DEBUG)
        extended_handler.setFormatter(log_format)
        self.logger.addHandler(extended_handler)
        self.log_handlers.append(extended_handler)

        # Handler 2: Basic log (INFO level - podstawowe informacje)
        basic_log_path = logs_dir / f"inference_{self.run_id}.log"
        basic_handler = logging.FileHandler(basic_log_path, mode='w', encoding='utf-8')
        basic_handler.setLevel(logging.INFO)
        basic_handler.setFormatter(log_format)
        self.logger.addHandler(basic_handler)
        self.log_handlers.append(basic_handler)

        # Ustaw poziom loggera na DEBUG żeby extended handler mógł wszystko złapać
        self.logger.setLevel(logging.DEBUG)

        self.logger.info(f"[LOGGING] Pliki logów skonfigurowane:")
        self.logger.info(f"  - Extended: {extended_log_path}")
        self.logger.info(f"  - Basic: {basic_log_path}")

    def _cleanup_logging(self):
        """
        Zamyka i usuwa handlery plików logów.
        Powinno być wywołane na końcu run() aby zapewnić poprawne zapisanie logów.
        """
        for handler in self.log_handlers:
            handler.flush()  # Upewnij się że wszystko jest zapisane
            handler.close()
            self.logger.removeHandler(handler)
        self.log_handlers.clear()
        self.logger.debug("[LOGGING] Handlery plików logów zamknięte")

    def run(self, df: pd.DataFrame, initial_facts: Optional[Set[Fact]] = None, dataset_name: str = "dataset") -> InferenceResult:
        """
        Uruchamia pełny pipeline eksperymentu.

        Args:
            df: DataFrame z danymi wejściowymi
            initial_facts: Opcjonalny zbiór faktów początkowych. Jeśli None,
                          utworzy fakty z pierwszego wiersza DataFrame.
            dataset_name: Nazwa datasetu (do zapisu artefaktów) - domyślnie "dataset"

        Returns:
            InferenceResult z wynikami wnioskowania

        Raises:
            ValueError: Gdy dane są niepoprawne lub brakuje wymaganej kolumny
        """
      
        self.run_id = self._generate_run_id()

        # Konfiguruj logowanie do plików (MUSI być po wygenerowaniu run_id!)
        self._setup_logging()

        self.logger.info("="*70)
        self.logger.info("EXPERIMENT START")
        self.logger.info("="*70)
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Configuration: seed={self.config.seed}, "
                        f"strategy={self.config.strategy.value}, "
                        f"method={self.config.generate_method.value}")

        # KROK 0: GLOBAL SEED (KRYTYCZNE!)
        self._set_global_seed()

        # KROK 1: Walidacja datasetu (opcjonalnie)
        if not self.config.skip_validation:
            self._validate_dataset(df)

        # KROK 2: Dyskretyzacja
        self._discretize_data(df)

        # KROK 3: Generowanie reguł
        self._generate_rules()

        # KROK 4: Klasteryzacja (opcjonalnie)
        if self.config.clustering_enabled:
            self._cluster_rules()

        # KROK 5: Wnioskowanie
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
                # Nie przerywamy - eksperyment się udał, tylko zapis nie poszedł

      
        if experiment_dir and FIREBASE_AVAILABLE:
            try:
                firebase = FirebaseService()

                # Sprawdź czy użytkownik jest zalogowany
                if firebase.is_logged_in():
                    self.logger.info("[FIREBASE] Rozpoczynam synchronizację z chmurą...")

                    # Upload folderu eksperymentu
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
                # Kontynuujemy działanie normalnie - lokalny zapis się udał

        # Zamknij handlery plików logów (upewnij się że wszystko jest zapisane)
        self._cleanup_logging()

        return result

    def _set_global_seed(self):
        """
        Ustawia GLOBAL SEED dla reprodukowalności.

        KRYTYCZNE: Musi być wywołane przed jakimkolwiek losowaniem!
        """
        self.logger.info(f"[SEED] Ustawianie GLOBAL_SEED={self.config.seed}")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.logger.info("[SEED] GLOBAL_SEED ustawiony - reprodukowalność zapewniona")

    def _validate_dataset(self, df: pd.DataFrame):
        """
        Waliduje dataset przed użyciem.

        Args:
            df: DataFrame do walidacji

        Raises:
            ValueError: Gdy dataset nie przejdzie walidacji krytycznej
        """
        self.logger.info("[VALIDATION] Starting dataset validation...")
        self.validator = DatasetReadinessValidator()
        report = self.validator.validate(df, decision_column=self.config.decision_column)

        self.logger.info(f"[VALIDATION] Score: {report.score}/100, Verdict: {report.verdict}")

        # Sprawdź czy są krytyczne problemy
        critical_issues = report.get_critical_issues()
        if critical_issues:
            self.logger.error(f"[VALIDATION] Found {len(critical_issues)} critical issues:")
            for issue in critical_issues:
                self.logger.error(f"  - {issue.message}")
            raise ValueError(f"Dataset failed validation: {len(critical_issues)} critical issues")

        self.logger.info("[VALIDATION] Dataset valid")

    def _discretize_data(self, df: pd.DataFrame):
        """
        Dyskretyzuje dane ciągłe.

        Args:
            df: DataFrame z danymi wejściowymi
        """
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
        """
        Generuje reguły z dyskretyzowanych danych.

        Raises:
            ValueError: Gdy kolumna decyzyjna nie istnieje
        """
        self.logger.info(f"[RULES] Method: {self.config.generate_method.value}")

        # Sprawdź czy kolumna decyzyjna istnieje
        if self.config.decision_column not in self.discretized_df.columns:
            raise ValueError(f"Decision column '{self.config.decision_column}' does not exist in DataFrame")

        # Safety net: kolumny ID powinny być usunięte już w csv_loader.load_csv(),
        # ale sprawdzamy ponownie na wypadek gdyby dane trafiły inną drogą (np. Firebase)
        temp_generator = RuleGenerator()
        id_columns = temp_generator.detect_id_columns(self.discretized_df)

        # Usuń kolumnę decyzyjną z listy ID (jeśli przypadkowo wykryta)
        id_columns = [col for col in id_columns if col != self.config.decision_column]

        if id_columns:
            self.logger.info(f"[RULES] Automatically detected and removed ID columns: {id_columns}")
            self.discretized_df = self.discretized_df.drop(columns=id_columns)

        # Wybierz generator na podstawie metody
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

        # Loguj statystyki reguł
        avg_premises = sum(len(r.premises) for r in self.generated_rules) / len(self.generated_rules) if self.generated_rules else 0
        self.logger.info(f"[RULES] Generated {len(self.generated_rules)} rules")
        self.logger.info(f"[RULES] Average number of premises: {avg_premises:.1f}")

        # Pokaż przykładowe reguły
        if self.generated_rules:
            self.logger.info("[RULES] Example rules:")
            for i, rule in enumerate(self.generated_rules[:3], 1):
                self.logger.info(f"  {i}. {rule}")

    def _cluster_rules(self):
        """
        Klasteryzuje reguły dla optymalizacji.
        """
        self.logger.info(f"[CLUSTERING] Number of clusters: {self.config.n_clusters}")

        # Dostosuj liczbę klastrów jeśli jest więcej niż reguł
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

        # Statystyki klastrów
        cluster_sizes = [len(cluster) for cluster in self.clusters]
        self.logger.info(f"[CLUSTERING] Cluster sizes: min={min(cluster_sizes)}, "
                        f"max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")

    def _run_inference(self, initial_facts: Optional[Set[Fact]]) -> InferenceResult:
        """
        Uruchamia silnik wnioskowania.

        Args:
            initial_facts: Opcjonalny zbiór faktów początkowych

        Returns:
            InferenceResult z wynikami wnioskowania
        """
        self.logger.info(f"[INFERENCE] Method: {self.config.inference_method.value}")

        # Stwórz fakty początkowe jeśli nie podano
        if initial_facts is None:
            initial_facts = self._create_facts_from_row(self.discretized_df.iloc[0])
            self.logger.info(f"[INFERENCE] Created {len(initial_facts)} facts from first row")
        else:
            self.logger.info(f"[INFERENCE] Used {len(initial_facts)} initial facts")

        # Stwórz KnowledgeBase
        kb = KnowledgeBase(rules=self.generated_rules, facts=initial_facts)

        # Stwórz strategię
        strategy = self._create_strategy()

        # Stwórz cel jeśli podano
        # Obsługuje zarówno tuple (atrybut, wartość) jak i string (nazwa atrybutu dla "Any Value")
        goal_target = None
        if self.config.goal is not None:
            if isinstance(self.config.goal, str):
                # Goal jako string - zatrzymaj gdy znajdzie DOWOLNĄ wartość tego atrybutu
                goal_target = self.config.goal
                self.logger.info(f"[INFERENCE] Goal: any value of attribute '{goal_target}'")
            else:
                # Goal jako tuple - zatrzymaj gdy znajdzie konkretny fakt
                goal_target = Fact(self.config.goal[0], str(self.config.goal[1]))
                self.logger.info(f"[INFERENCE] Goal: {goal_target}")

        # Inicjalizuj silnik na podstawie konfiguracji
        if self.config.inference_method == InferenceMethod.BACKWARD:
            # Backward Chaining - wymaga konkretnego celu (Fact), nie obsługuje "Any Value" (string)
            if goal_target is None:
                raise ValueError("Backward Chaining requires a goal")
            if isinstance(goal_target, str):
                raise ValueError("Backward Chaining requires a specific goal (attribute + value), "
                               "'Any Value' option is not supported for Backward Chaining")

            self.inference_engine = BackwardChaining(strategy=strategy)
            result = self.inference_engine.run(kb, goal=goal_target)

        elif self.config.inference_method == InferenceMethod.GREEDY:
            # Greedy Forward Chaining
            self.inference_engine = GreedyForwardChaining()
            result = self.inference_engine.run(kb, goal=goal_target)

        else:  # FORWARD
            if self.config.clustering_enabled and self.clusters is not None:
                # Clustered Forward Chaining
                self.logger.info("[INFERENCE] Using ClusteredForwardChaining")
                self.inference_engine = ClusteredForwardChaining(
                    strategy=strategy,
                    clusters=self.clusters,
                    centroid_match_threshold=self.config.centroid_match_threshold
                )
                result = self.inference_engine.run(kb, goal=goal_target)
            else:
                # Standardowy Forward Chaining
                self.inference_engine = ForwardChaining(strategy=strategy)
                result = self.inference_engine.run(kb, goal=goal_target, greedy=False)

        self.logger.info(f"[INFERENCE] Completed. Success: {result.success}, "
                        f"New facts: {len(result.new_facts)}, "
                        f"Iterations: {result.iterations}")

        return result

    def _create_strategy(self) -> ConflictResolutionStrategy:
        """
        Tworzy strategię conflict resolution.

        Returns:
            Instancja strategii
        """
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
        """
        Tworzy zbiór faktów z wiersza DataFrame.

        Args:
            row: Wiersz pandas DataFrame

        Returns:
            Zbiór faktów
        """
        facts = set()

        for column, value in row.items():
            # Pomiń NaN
            if pd.isna(value):
                continue

            # Konwertuj wartość na string
            value_str = str(value)

            # Stwórz Fact
            fact = Fact(attribute=column, value=value_str)
            facts.add(fact)

        return facts

    def _generate_run_id(self) -> str:
        """
        Generuje unikalny identyfikator eksperymentu.
        Np. "exp_20240115_123045"

        Returns:
            Unikalny run_id oparty na timestampie
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"exp_{timestamp}"
        return run_id
