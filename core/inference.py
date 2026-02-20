"""
Moduł zawierający silniki wnioskowania w przód i wstecz.

Implementacja zgodna z Algorithm 1 z artykułu.

Klasy:
    - InferenceResult: wynik procesu wnioskowania
    - ForwardChaining: klasyczny silnik z użyciem strategii conflict resolution
    - GreedyForwardChaining: zachłanny silnik (odpala wszystkie reguły naraz)
    - BackwardChaining: silnik wnioskowania wstecz (goal-driven)
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Union
import time
import logging

from core.models import Fact, Rule, KnowledgeBase
from core.strategies import ConflictResolutionStrategy, RecencyStrategy

# Domyślny logger dla modułu (fallback)
default_logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """
    Wynik procesu wnioskowania.

    Attributes:
        success: Czy cel został osiągnięty (dla goal-driven) lub True dla classic
        facts: Końcowy zbiór wszystkich faktów
        new_facts: Lista nowych faktów dodanych podczas wnioskowania (w kolejności)
        rules_fired: Lista reguł które zostały odpalone (w kolejności)
        iterations: Liczba iteracji głównej pętli
        execution_time_ms: Czas wykonania w milisekundach
        rules_evaluated: Liczba sprawdzeń warunków IF reguł (ile razy wywołano is_satisfied_by)
        rules_activated: Suma rozmiarów zbiorów konfliktowych (conflict sets) ze wszystkich iteracji
        facts_count: Liczba faktów w bazie na koniec wnioskowania
        trace: Szczegółowy ślad wnioskowania (Deep Investigation)
    """
    success: bool
    facts: Set[Fact]
    new_facts: List[Fact]
    rules_fired: List[Rule]
    iterations: int
    execution_time_ms: float
    rules_evaluated: int
    rules_activated: int
    facts_count: int
    trace: List[str] = field(default_factory=list)


class ForwardChaining:
    """
    Silnik wnioskowania w przód (Forward Chaining).

    Implementacja zgodna z Algorithm 1 z artykułu.
    Używa strategii conflict resolution do wyboru reguły z conflict set.

    Example:
        >>> strategy = FirstStrategy()
        >>> engine = ForwardChaining(strategy)
        >>> result = engine.run(kb, goal=Fact("diagnoza", "grypa"))
    """

    def __init__(self, strategy: ConflictResolutionStrategy, run_id: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Tworzy silnik wnioskowania.

        Args:
            strategy: Strategia wyboru reguły z conflict set
            run_id: Unikalny identyfikator uruchomienia (opcjonalny, dla logów)
            logger: Dedykowany logger (opcjonalny)
        """
        self.strategy = strategy
        self.run_id = run_id
        self.logger = logger if logger else default_logger

    def run(self, kb: KnowledgeBase, goal: Optional[Union[Fact, str]] = None, greedy: bool = False) -> InferenceResult:
        """
        Uruchamia wnioskowanie w przód.

        Algorytm (z artykułu):
        1. Dopóki są reguły do odpalenia:
           a. Zbuduj conflict set (reguły z spełnionymi przesłankami i niespełnioną konkluzją)
           b. Jeśli conflict set pusty - koniec
           c. GREEDY MODE: Odpal wszystkie reguły z conflict set
              NORMAL MODE: Wybierz regułę używając strategii
           d. Dodaj konkluzję do faktów
           e. Jeśli cel osiągnięty - zatrzymaj (SUCCESS)
        2. Jeśli cel nie osiągnięty - FAILURE, w przeciwnym razie SUCCESS

        Args:
            kb: Baza wiedzy z regułami i faktami początkowymi
            goal: Opcjonalny cel - może być:
                  - Fact: zatrzymaj gdy konkretny fakt zostanie wywnioskowany
                  - str: zatrzymaj gdy DOWOLNA wartość atrybutu o tej nazwie zostanie znaleziona
                  - None: wnioskowanie pełne (do wyczerpania)
            greedy: Jeśli True, odpala wszystkie reguły z conflict set (bez strategii)

        Returns:
            InferenceResult z wynikami wnioskowania i metrykami wydajnościowymi
        """
        start_time = time.perf_counter()

        # Kopiujemy fakty aby nie modyfikować oryginalnej bazy
        facts = kb.facts.copy()
        new_facts = []
        rules_fired = []
        iterations = 0

        # REFRACTORINESS: Zbiór ID reguł które już odpaliły
        # Zgodnie z conflictSet.pdf str. 3 - reguła może odpalić tylko raz
        fired_rules_ids: Set[int] = set()

        # DEEP INVESTIGATION: Trace dla szczegółowego śledzenia
        trace: List[str] = []

        # Metryki wydajnościowe
        rules_evaluated = 0  # Licznik wywołań is_satisfied_by()
        rules_activated = 0   # Suma rozmiarów conflict sets

        # Logical Clock for RecencyStrategy (iteration-based, not time-based)
        # Initial facts have logical_clock_id = 0
        # New facts get logical_clock_id = current iteration number (1, 2, 3, ...)
        facts_with_recency: Dict[Fact, int] = {fact: 0 for fact in facts}
        is_recency_strategy = isinstance(self.strategy, RecencyStrategy)

        # Helper: Check if goal is achieved (supports both Fact and attribute-name string)
        def _goal_achieved(current_facts: Set[Fact], goal_target) -> bool:
            if goal_target is None:
                return False
            if isinstance(goal_target, str):
                # Goal is attribute name - stop when ANY value for this attribute is found
                return any(f.attribute == goal_target for f in current_facts)
            else:
                # Goal is a Fact - stop when this exact fact is found
                return goal_target in current_facts

        mode_info = "[GREEDY MODE]" if greedy else f"[Strategy: {self.strategy.__class__.__name__}]"
        self.logger.info(f"=== Starting Forward Chaining Inference {mode_info} {f'(Run ID: {self.run_id})' if self.run_id else ''} ===")

        # XAI: Extended log header with KB summary
        self.logger.debug(f"Inference started. Knowledge Base contains {len(kb.rules)} rules and {len(facts)} initial facts.")

        self.logger.info(f"Initial facts: {len(facts)}, Rules: {len(kb.rules)}, Goal: {goal}")

        inferred = True
        while inferred:
            iterations += 1
            inferred = False

            self.logger.info(f"[STEP {iterations}] Current facts count: {len(facts)}")

            # Budujemy conflict set - reguły gotowe do odpalenia
            # REFRACTORINESS: Reguła może odpalić tylko raz (zgodnie z conflictSet.pdf str. 3)
            conflict_set = []
            for rule in kb.rules:
                rules_evaluated += 1  # Zliczamy każde sprawdzenie warunku
                if (rule.id not in fired_rules_ids
                    and rule.is_satisfied_by(facts)
                    and rule.conclusion not in facts):  # Algorithm 1: r.conclusion ∉ FB
                    conflict_set.append(rule)

            # Aktualizujemy metryki
            rules_activated += len(conflict_set)

            # ========== DEEP INVESTIGATION: Build trace for this iteration ==========
            trace_lines = []
            trace_lines.append("=" * 80)
            trace_lines.append(f"ITERATION {iterations}")
            trace_lines.append("=" * 80)

            # Current facts
            trace_lines.append(f"\nCURRENT FACTS ({len(facts)}):")
            for fact in sorted(facts, key=lambda f: f.attribute):
                trace_lines.append(f"  - {fact.attribute}={fact.value}")

            # XAI: Log conflict set with rule IDs for explainability
            if conflict_set:
                rule_ids = [r.id for r in conflict_set]
                self.logger.info(f"[STEP {iterations}] Conflict set size: {len(conflict_set)} rules. Candidates: {rule_ids}")

                # Trace: Conflict set
                trace_lines.append(f"\nCONFLICT SET ({len(conflict_set)} rules applicable):")
                for rule in conflict_set:
                    premises_text = " AND ".join([f"{p.attribute}={p.value}" for p in rule.premises])
                    conclusion_text = f"{rule.conclusion.attribute}={rule.conclusion.value}"
                    trace_lines.append(f"  [Rule {rule.id}] IF ({premises_text}) THEN {conclusion_text}")

                # EXTENDED XAI: Log full rule details to extended log (DEBUG level)
                self.logger.debug(f"[STEP {iterations}] Conflict Set Details:")
                for rule in conflict_set:
                    # Build premises text
                    premises_text = " AND ".join([f"{p.attribute}={p.value}" for p in rule.premises])
                    conclusion_text = f"{rule.conclusion.attribute}={rule.conclusion.value}"
                    complexity = len(rule.premises)

                    debug_msg = f"  [CANDIDATE] Rule {rule.id}: IF {premises_text} THEN {conclusion_text} | Complexity: {complexity}"

                    # Add Logical Clock if using RecencyStrategy
                    if is_recency_strategy:
                        # Calculate max logical clock ID from premises
                        max_clock_id = max([facts_with_recency.get(p, 0) for p in rule.premises])
                        debug_msg += f" | Max Logical Clock: {max_clock_id}"

                    self.logger.debug(debug_msg)
            else:
                self.logger.info(f"[STEP {iterations}] Conflict set EMPTY. No applicable rules found. Stopping.")
                trace_lines.append(f"\nCONFLICT SET (0 rules applicable):")
                trace_lines.append("  (empty - no rules can fire)")
                trace_lines.append(f"\nACTION:")
                trace_lines.append("  Inference stopped - no applicable rules.")
                trace.append("\n".join(trace_lines))

            # Jeśli jest jakaś reguła do odpalenia
            if conflict_set:
                if greedy:
                    # ========== TRYB GREEDY: Odpal WSZYSTKIE reguły z conflict set ==========
                    fired_count = 0
                    new_facts_this_iteration = []

                    for rule in conflict_set:
                        # Sprawdzamy redundancję (czy fakt już nie istnieje)
                        if rule.conclusion not in facts:
                            facts.add(rule.conclusion)
                            new_facts.append(rule.conclusion)
                            rules_fired.append(rule)
                            new_facts_this_iteration.append(rule.conclusion)
                            fired_count += 1

                            # REFRACTORINESS: Oznacz regułę jako odpaloną
                            fired_rules_ids.add(rule.id)

                            # Update logical clock for new fact (if using RecencyStrategy)
                            if is_recency_strategy:
                                facts_with_recency[rule.conclusion] = iterations  # Logical Clock ID

                    # DEEP INVESTIGATION: Greedy mode trace
                    trace_lines.append(f"\nSTRATEGY DECISION:")
                    trace_lines.append(f"  Strategy: GREEDY MODE")
                    trace_lines.append(f"  Action: Fire ALL {len(conflict_set)} rules in parallel")

                    trace_lines.append(f"\nACTION:")
                    if fired_count > 0:
                        self.logger.info(f"[GREEDY] Firing {fired_count} rules in parallel. New facts: {new_facts_this_iteration}")
                        trace_lines.append(f"  Rules Fired: {fired_count}")
                        trace_lines.append(f"  New Facts Added:")
                        for nf in new_facts_this_iteration:
                            trace_lines.append(f"    - {nf.attribute}={nf.value}")
                        inferred = True
                    else:
                        self.logger.info(f"[GREEDY] All {len(conflict_set)} rules provided NO new facts (redundant).")
                        trace_lines.append(f"  No new facts (all conclusions already known)")

                    trace.append("\n".join(trace_lines))

                else:
                    # ========== TRYB NORMAL: Wybierz jedną regułę strategią ==========
                    if is_recency_strategy:
                        # RecencyStrategy potrzebuje Dict[Fact, int]
                        selected_rule = self.strategy.select(conflict_set, facts_with_recency)
                    else:
                        # Inne strategie używają Set[Fact]
                        selected_rule = self.strategy.select(conflict_set, facts)

                    self.logger.info(f"[STRATEGY] {self.strategy.__class__.__name__} selected Rule {selected_rule.id}: {selected_rule}")

                    # DEEP INVESTIGATION: Strategy decision trace
                    selected_premises = " AND ".join([f"{p.attribute}={p.value}" for p in selected_rule.premises])
                    selected_conclusion = f"{selected_rule.conclusion.attribute}={selected_rule.conclusion.value}"
                    trace_lines.append(f"\nSTRATEGY DECISION:")
                    trace_lines.append(f"  Strategy: {self.strategy.__class__.__name__}")
                    trace_lines.append(f"  Selected Rule: [Rule {selected_rule.id}] IF ({selected_premises}) THEN {selected_conclusion}")

                    # Add reason based on strategy type
                    strategy_name = self.strategy.__class__.__name__
                    if strategy_name == "SpecificityStrategy":
                        trace_lines.append(f"  Reason: Longest rule with {len(selected_rule.premises)} premises")
                    elif strategy_name == "RecencyStrategy":
                        max_clock = max([facts_with_recency.get(p, 0) for p in selected_rule.premises])
                        trace_lines.append(f"  Reason: Uses newest facts (max logical clock: {max_clock})")
                    elif strategy_name == "RandomStrategy":
                        trace_lines.append(f"  Reason: Random selection from {len(conflict_set)} candidates")
                    elif strategy_name == "FirstStrategy":
                        trace_lines.append(f"  Reason: First rule in conflict set (FIFO)")

                    # Sprawdzamy redundancję
                    if selected_rule.conclusion not in facts:
                        new_facts_from_rule = [selected_rule.conclusion]

                        # Dodajemy konkluzję do faktów
                        facts.add(selected_rule.conclusion)
                        new_facts.append(selected_rule.conclusion)
                        rules_fired.append(selected_rule)

                        # REFRACTORINESS: Oznacz regułę jako odpaloną
                        fired_rules_ids.add(selected_rule.id)

                        self.logger.info(f"[FIRE] Rule {selected_rule.id} fired! New facts inferred: {new_facts_from_rule}")

                        # DEEP INVESTIGATION: Action trace
                        trace_lines.append(f"\nACTION:")
                        trace_lines.append(f"  Rule Fired!")
                        trace_lines.append(f"  New Fact Added: {selected_rule.conclusion.attribute}={selected_rule.conclusion.value}")

                        # Update logical clock for new fact
                        if is_recency_strategy:
                            facts_with_recency[selected_rule.conclusion] = iterations  # Logical Clock ID

                        inferred = True
                    else:
                        self.logger.info(f"[SKIP] Rule {selected_rule.id} activated but provided NO new facts.")
                        trace_lines.append(f"\nACTION:")
                        trace_lines.append(f"  Rule SKIPPED (conclusion already known)")

                    # Add this iteration's trace to the main trace
                    trace.append("\n".join(trace_lines))

                # Sprawdzamy czy osiągnęliśmy cel (działa dla obu trybów)
                # Obsługuje zarówno Fact (konkretna wartość) jak i str (dowolna wartość atrybutu)
                if _goal_achieved(facts, goal):
                    end_time = time.perf_counter()
                    execution_time_ms = (end_time - start_time) * 1000

                    self.logger.info(f"=== Goal {goal} ACHIEVED in iteration {iterations} ===")
                    self.logger.info(f"Execution time: {execution_time_ms:.3f} ms")

                    # Add final summary to trace
                    trace.append("\n" + "=" * 80)
                    trace.append(f"INFERENCE COMPLETED - GOAL ACHIEVED")
                    trace.append(f"Total iterations: {iterations}, Rules fired: {len(rules_fired)}")
                    trace.append("=" * 80)

                    return InferenceResult(
                        success=True,
                        facts=facts,
                        new_facts=new_facts,
                        rules_fired=rules_fired,
                        iterations=iterations,
                        execution_time_ms=execution_time_ms,
                        rules_evaluated=rules_evaluated,
                        rules_activated=rules_activated,
                        facts_count=len(facts),
                        trace=trace
                    )

        # Koniec wnioskowania
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Jeśli był cel i nie osiągnęliśmy go - FAILURE
        # (sprawdzamy jeszcze raz na wypadek gdyby cel był w initial facts)
        if goal is not None:
            if _goal_achieved(facts, goal):
                success = True
                self.logger.info(f"=== Goal {goal} ACHIEVED (found in final facts) ===")
            else:
                success = False
                self.logger.info(f"=== Goal {goal} NOT ACHIEVED ===")
        else:
            success = True
            self.logger.info(f"=== Inference COMPLETED ===")

        self.logger.info(f"Total iterations: {iterations}, Facts: {len(facts)}, Rules fired: {len(rules_fired)}")
        self.logger.info(f"Execution time: {execution_time_ms:.3f} ms")
        self.logger.info(f"Rules evaluated: {rules_evaluated}, Rules activated: {rules_activated}")

        # Add final summary to trace
        trace.append("\n" + "=" * 80)
        if goal is not None and not success:
            trace.append(f"INFERENCE COMPLETED - GOAL NOT ACHIEVED")
        else:
            trace.append(f"INFERENCE COMPLETED")
        trace.append(f"Total iterations: {iterations}, Rules fired: {len(rules_fired)}, Final facts: {len(facts)}")
        trace.append("=" * 80)

        return InferenceResult(
            success=success,
            facts=facts,
            new_facts=new_facts,
            rules_fired=rules_fired,
            iterations=iterations,
            execution_time_ms=execution_time_ms,
            rules_evaluated=rules_evaluated,
            rules_activated=rules_activated,
            facts_count=len(facts),
            trace=trace
        )


class GreedyForwardChaining:
    """
    Zachłanny silnik wnioskowania w przód (Greedy Forward Chaining).

    W przeciwieństwie do klasycznego Forward Chaining, ten silnik aktywuje
    WSZYSTKIE pasujące reguły w każdej iteracji, zamiast wybierać jedną.
    Nie używa strategii conflict resolution.

    Example:
        >>> engine = GreedyForwardChaining()
        >>> result = engine.run(kb)
    """

    def run(self, kb: KnowledgeBase, goal: Optional[Union[Fact, str]] = None) -> InferenceResult:
        """
        Uruchamia zachłanne wnioskowanie w przód.

        Aktywuje wszystkie pasujące reguły w każdej iteracji.

        Args:
            kb: Baza wiedzy z regułami i faktami początkowymi
            goal: Opcjonalny cel - może być:
                  - Fact: zatrzymaj gdy konkretny fakt zostanie wywnioskowany
                  - str: zatrzymaj gdy DOWOLNA wartość atrybutu o tej nazwie zostanie znaleziona
                  - None: wnioskowanie pełne (do wyczerpania)

        Returns:
            InferenceResult z wynikami wnioskowania i metrykami wydajnościowymi
        """
        start_time = time.perf_counter()

        # Kopiujemy fakty aby nie modyfikować oryginalnej bazy
        facts = kb.facts.copy()
        new_facts = []
        rules_fired = []
        iterations = 0

        # Metryki wydajnościowe
        rules_evaluated = 0
        rules_activated = 0

        # Helper: Check if goal is achieved (supports both Fact and attribute-name string)
        def _goal_achieved(current_facts: Set[Fact], goal_target) -> bool:
            if goal_target is None:
                return False
            if isinstance(goal_target, str):
                return any(f.attribute == goal_target for f in current_facts)
            else:
                return goal_target in current_facts

        default_logger.info(f"=== Starting Greedy Forward Chaining Inference ===")
        default_logger.info(f"Initial facts: {len(facts)}, Rules: {len(kb.rules)}, Goal: {goal}")

        inferred = True
        while inferred:
            iterations += 1
            inferred = False

            default_logger.debug(f"--- Iteration {iterations} ---")

            # Budujemy conflict set - wszystkie reguły gotowe do odpalenia
            conflict_set = []
            for rule in kb.rules:
                rules_evaluated += 1
                if rule.is_satisfied_by(facts) and rule.conclusion not in facts:
                    conflict_set.append(rule)

            rules_activated += len(conflict_set)

            default_logger.debug(f"Conflict Set size: {len(conflict_set)}, Rule IDs: {[r.id for r in conflict_set]}")

            # Odpalamy WSZYSTKIE reguły z conflict set
            if conflict_set:
                for rule in conflict_set:
                    facts.add(rule.conclusion)
                    new_facts.append(rule.conclusion)
                    rules_fired.append(rule)
                    default_logger.info(f"Fired Rule {rule.id}, New Fact: {rule.conclusion}")

                inferred = True

                # Sprawdzamy czy osiągnęliśmy cel
                # Obsługuje zarówno Fact (konkretna wartość) jak i str (dowolna wartość atrybutu)
                if _goal_achieved(facts, goal):
                    end_time = time.perf_counter()
                    execution_time_ms = (end_time - start_time) * 1000

                    default_logger.info(f"=== Goal {goal} ACHIEVED in iteration {iterations} ===")

                    return InferenceResult(
                        success=True,
                        facts=facts,
                        new_facts=new_facts,
                        rules_fired=rules_fired,
                        iterations=iterations,
                        execution_time_ms=execution_time_ms,
                        rules_evaluated=rules_evaluated,
                        rules_activated=rules_activated,
                        facts_count=len(facts)
                    )

        # Koniec wnioskowania
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Jeśli był cel i nie osiągnęliśmy go - FAILURE
        # (sprawdzamy jeszcze raz na wypadek gdyby cel był w initial facts)
        if goal is not None:
            if _goal_achieved(facts, goal):
                success = True
                default_logger.info(f"=== Goal {goal} ACHIEVED (found in final facts) ===")
            else:
                success = False
                default_logger.info(f"=== Goal {goal} NOT ACHIEVED ===")
        else:
            success = True
            default_logger.info(f"=== Inference COMPLETED ===")

        default_logger.info(f"Total iterations: {iterations}, Facts: {len(facts)}, Rules fired: {len(rules_fired)}")
        default_logger.info(f"Rules evaluated: {rules_evaluated}, Rules activated: {rules_activated}")

        return InferenceResult(
            success=success,
            facts=facts,
            new_facts=new_facts,
            rules_fired=rules_fired,
            iterations=iterations,
            execution_time_ms=execution_time_ms,
            rules_evaluated=rules_evaluated,
            rules_activated=rules_activated,
            facts_count=len(facts)
        )


class BackwardChaining:
    """
    Silnik wnioskowania wstecz (Backward Chaining) - WERSJA Z DUAL LOGGING.

    Goal-driven: zaczyna od celu i próbuje go udowodnić
    znajdując reguły i rekurencyjnie udowadniając przesłanki.

    Używa backtrackingu - jeśli pierwsza reguła nie działa,
    próbuje następnej.

    XAI Features:
    - Standard Log (INFO): High-level proof steps
    - Extended Log (DEBUG): Full rule details, recursion trace, competitive rules

    Example:
        >>> strategy = SpecificityStrategy()
        >>> engine = BackwardChaining(strategy, run_id="test_001", logger=my_logger)
        >>> result = engine.run(kb, goal=Fact("diagnoza", "grypa"))
    """

    def __init__(self, strategy: ConflictResolutionStrategy, run_id: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Tworzy silnik wnioskowania wstecz.

        Args:
            strategy: Strategia wyboru reguły gdy wiele reguł może udowodnić ten sam cel
            run_id: Unikalny identyfikator uruchomienia (opcjonalny, dla logów)
            logger: Dedykowany logger (opcjonalny, dla dual logging)
        """
        self.strategy = strategy
        self.run_id = run_id
        self.logger = logger if logger else default_logger

    def run(self, kb: KnowledgeBase, goal: Fact) -> InferenceResult:
        """
        Uruchamia wnioskowanie wstecz z DUAL LOGGING.

        Algorytm:
        1. Sprawdź czy cel już w faktach -> Sukces
        2. Znajdź reguły konkurencyjne (konkluzja = cel)
        3. Użyj strategii do wyboru kolejności
        4. Dla każdej reguły próbuj udowodnić wszystkie przesłanki (rekurencja)
        5. Jeśli wszystkie udowodnione - SUCCESS, dodaj cel do faktów
        6. Jeśli nie - backtrack do następnej reguły

        Args:
            kb: Baza wiedzy z regułami i faktami początkowymi
            goal: Cel do udowodnienia (WYMAGANY)

        Returns:
            InferenceResult z wynikami wnioskowania i metrykami wydajnościowymi
        """
        start_time = time.perf_counter()

        # Kopiujemy fakty aby nie modyfikować oryginalnej bazy
        facts = kb.facts.copy()
        new_facts = []
        rules_fired = []

        # Zbiór celów aktualnie przetwarzanych (wykrywanie cykli)
        proof_path = set()

        # Metryki wydajnościowe
        metrics = {
            'rules_evaluated': 0,
            'rules_activated': 0,
            'recursion_depth': 0,
            'max_depth': 0
        }

        self.logger.info(f"=== Starting Backward Chaining Inference {f'(Run ID: {self.run_id})' if self.run_id else ''} ===")

        # XAI: Extended log header with KB summary
        self.logger.debug(f"Inference started. Knowledge Base contains {len(kb.rules)} rules and {len(facts)} initial facts.")

        self.logger.info(f"Initial facts: {len(facts)}, Rules: {len(kb.rules)}, Goal: {goal}")

        # Próbujemy udowodnić cel
        success = self._prove(
            goal=goal,
            rules=kb.rules,
            facts=facts,
            new_facts=new_facts,
            rules_fired=rules_fired,
            proof_path=proof_path,
            metrics=metrics,
            depth=0
        )

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        if success:
            self.logger.info(f"=== Goal {goal} PROVED ===")
        else:
            self.logger.info(f"=== Goal {goal} FAILED (not provable) ===")

        self.logger.info(f"Rules fired: {len(rules_fired)}, New facts: {len(new_facts)}, Total facts: {len(facts)}")
        self.logger.info(f"Execution time: {execution_time_ms:.3f} ms")
        self.logger.info(f"Rules evaluated: {metrics['rules_evaluated']}, Rules activated: {metrics['rules_activated']}")
        self.logger.debug(f"Max recursion depth reached: {metrics['max_depth']}")

        return InferenceResult(
            success=success,
            facts=facts,
            new_facts=new_facts,
            rules_fired=rules_fired,
            iterations=len(rules_fired),  # liczba odpalonych reguł
            execution_time_ms=execution_time_ms,
            rules_evaluated=metrics['rules_evaluated'],
            rules_activated=metrics['rules_activated'],
            facts_count=len(facts)
        )

    def _prove(
        self,
        goal: Fact,
        rules: List[Rule],
        facts: Set[Fact],
        new_facts: List[Fact],
        rules_fired: List[Rule],
        proof_path: Set[Fact],
        metrics: Dict[str, int],
        depth: int
    ) -> bool:
        """
        Rekurencyjna metoda próbująca udowodnić cel z DUAL LOGGING.

        Args:
            goal: Cel do udowodnienia
            rules: Lista wszystkich reguł
            facts: Aktualny zbiór faktów
            new_facts: Lista nowych faktów (mutowana)
            rules_fired: Lista odpalonych reguł (mutowana)
            proof_path: Cele w trakcie przetwarzania (wykrywanie cykli)
            metrics: Słownik metryk wydajnościowych (mutowany)
            depth: Głębokość rekurencji (dla logowania)

        Returns:
            True jeśli cel udowodniony, False w przeciwnym razie
        """
        # Track max recursion depth
        metrics['max_depth'] = max(metrics['max_depth'], depth)
        indent = "  " * depth  # Indentation for readability

        # Krok 1: Czy cel już w faktach?
        if goal in facts:
            self.logger.debug(f"{indent}[DEPTH {depth}] Goal {goal} already in facts (base case)")
            return True

        # Krok 2: Wykrywanie cykli - czy ten cel jest już przetwarzany?
        if goal in proof_path:
            self.logger.debug(f"{indent}[DEPTH {depth}] CYCLE DETECTED for goal {goal} - stopping recursion")
            self.logger.info(f"[BACKWARD] Cycle detected for {goal}, backtracking...")
            return False

        # Dodajemy cel do ścieżki dowodu
        proof_path.add(goal)

        self.logger.info(f"[BACKWARD] Trying to prove goal: {goal}")
        self.logger.debug(f"{indent}[DEPTH {depth}] Attempting to prove: {goal}")

        # Krok 3: Znajdź reguły konkurencyjne (konkluzja = cel)
        competitive_rules = []
        for rule in rules:
            metrics['rules_evaluated'] += 1
            if rule.conclusion == goal:
                competitive_rules.append(rule)

        if not competitive_rules:
            # Brak reguł do udowodnienia celu
            self.logger.debug(f"{indent}[DEPTH {depth}] No competitive rules for goal {goal} - FAIL")
            self.logger.info(f"[BACKWARD] No rules can prove {goal}")
            proof_path.remove(goal)
            return False

        metrics['rules_activated'] += len(competitive_rules)

        # STANDARD LOG: Rule IDs only
        rule_ids = [r.id for r in competitive_rules]
        self.logger.info(f"[BACKWARD] Found {len(competitive_rules)} competitive rules for {goal}: {rule_ids}")

        # EXTENDED LOG: Full rule details with complexity
        self.logger.debug(f"{indent}[DEPTH {depth}] Competitive Rules Details:")
        for rule in competitive_rules:
            premises_text = " AND ".join([f"{p.attribute}={p.value}" for p in rule.premises])
            conclusion_text = f"{rule.conclusion.attribute}={rule.conclusion.value}"
            complexity = len(rule.premises)
            self.logger.debug(f"{indent}  [CANDIDATE] Rule {rule.id}: IF {premises_text} THEN {conclusion_text} | Complexity: {complexity}")

        # Krok 4: Użyj strategii do wyboru kolejności reguł
        ordered_rules = self._order_rules_by_strategy(competitive_rules, facts)
        self.logger.debug(f"{indent}[DEPTH {depth}] Strategy ordered rules: {[r.id for r in ordered_rules]}")

        # Krok 5: Backtracking - próbuj każdą regułę po kolei
        for rule in ordered_rules:
            self.logger.debug(f"{indent}[DEPTH {depth}] Trying rule {rule.id} with {len(rule.premises)} premises")

            # Próbujemy udowodnić wszystkie przesłanki reguły
            all_premises_proven = True

            for i, premise in enumerate(rule.premises, 1):
                self.logger.debug(f"{indent}[DEPTH {depth}] Proving premise {i}/{len(rule.premises)}: {premise}")

                # Rekurencyjnie próbujemy udowodnić przesłankę
                if not self._prove(
                    goal=premise,
                    rules=rules,
                    facts=facts,
                    new_facts=new_facts,
                    rules_fired=rules_fired,
                    proof_path=proof_path,
                    metrics=metrics,
                    depth=depth + 1
                ):
                    # Ta przesłanka nie może być udowodniona
                    self.logger.debug(f"{indent}[DEPTH {depth}] Premise {premise} FAILED - cannot prove")
                    all_premises_proven = False
                    break  # nie ma sensu sprawdzać dalszych przesłanek

            # Jeśli wszystkie przesłanki udowodnione - sukces!
            if all_premises_proven:
                # Dodajemy cel do faktów
                facts.add(goal)
                new_facts.append(goal)
                rules_fired.append(rule)

                self.logger.info(f"[SUCCESS] Proved {goal} using Rule {rule.id}")
                self.logger.debug(f"{indent}[DEPTH {depth}] SUCCESS - All premises proven for rule {rule.id}")

                # Usuwamy cel ze ścieżki
                proof_path.remove(goal)
                return True

            # Ta reguła nie zadziałała, próbujemy następnej (backtracking)
            self.logger.debug(f"{indent}[DEPTH {depth}] Rule {rule.id} failed, backtracking to next candidate...")

        # Żadna reguła nie zadziałała
        self.logger.debug(f"{indent}[DEPTH {depth}] All {len(competitive_rules)} rules exhausted for {goal} - FAIL")
        self.logger.info(f"[BACKWARD] All rules failed for {goal}, cannot prove")
        proof_path.remove(goal)
        return False

    def _order_rules_by_strategy(
        self,
        rules: List[Rule],
        facts: Set[Fact]
    ) -> List[Rule]:
        """
        Uporządkuj reguły według strategii.

        Strategia select() wybiera jedną "najlepszą" regułę.
        Używamy jej iteracyjnie do stworzenia uporządkowanej listy.

        Args:
            rules: Lista reguł do uporządkowania
            facts: Aktualne fakty (dla strategii które ich potrzebują)

        Returns:
            Lista reguł w kolejności preferowanej przez strategię
        """
        if len(rules) == 1:
            return rules

        ordered = []
        remaining = rules.copy()

        while remaining:
            # Wybierz najlepszą regułę z pozostałych
            selected = self.strategy.select(remaining, facts)
            ordered.append(selected)
            remaining.remove(selected)

        return ordered