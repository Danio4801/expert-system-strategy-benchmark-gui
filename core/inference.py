











from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Union
import time
import logging

from core.models import Fact, Rule, KnowledgeBase
from core.strategies import ConflictResolutionStrategy, RecencyStrategy


default_logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:















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












    def __init__(self, strategy: ConflictResolutionStrategy, run_id: Optional[str] = None, logger: Optional[logging.Logger] = None):








        self.strategy = strategy
        self.run_id = run_id
        self.logger = logger if logger else default_logger

    def run(self, kb: KnowledgeBase, goal: Optional[Union[Fact, str]] = None, greedy: bool = False) -> InferenceResult:
























        start_time = time.perf_counter()


        facts = kb.facts.copy()
        new_facts = []
        rules_fired = []
        iterations = 0



        fired_rules_ids: Set[int] = set()


        trace: List[str] = []


        rules_evaluated = 0
        rules_activated = 0




        facts_with_recency: Dict[Fact, int] = {fact: 0 for fact in facts}
        is_recency_strategy = isinstance(self.strategy, RecencyStrategy)


        def _goal_achieved(current_facts: Set[Fact], goal_target) -> bool:
            if goal_target is None:
                return False
            if isinstance(goal_target, str):

                return any(f.attribute == goal_target for f in current_facts)
            else:

                return goal_target in current_facts

        mode_info = "[GREEDY MODE]" if greedy else f"[Strategy: {self.strategy.__class__.__name__}]"
        self.logger.info(f"=== Starting Forward Chaining Inference {mode_info} {f'(Run ID: {self.run_id})' if self.run_id else ''} ===")


        self.logger.debug(f"Inference started. Knowledge Base contains {len(kb.rules)} rules and {len(facts)} initial facts.")

        self.logger.info(f"Initial facts: {len(facts)}, Rules: {len(kb.rules)}, Goal: {goal}")

        inferred = True
        while inferred:
            iterations += 1
            inferred = False

            self.logger.info(f"[STEP {iterations}] Current facts count: {len(facts)}")



            conflict_set = []
            for rule in kb.rules:
                rules_evaluated += 1
                if (rule.id not in fired_rules_ids
                    and rule.is_satisfied_by(facts)
                    and rule.conclusion not in facts):
                    conflict_set.append(rule)


            rules_activated += len(conflict_set)


            trace_lines = []
            trace_lines.append("=" * 80)
            trace_lines.append(f"ITERATION {iterations}")
            trace_lines.append("=" * 80)


            trace_lines.append(f"\nCURRENT FACTS ({len(facts)}):")
            for fact in sorted(facts, key=lambda f: f.attribute):
                trace_lines.append(f"  - {fact.attribute}={fact.value}")


            if conflict_set:
                rule_ids = [r.id for r in conflict_set]
                self.logger.info(f"[STEP {iterations}] Conflict set size: {len(conflict_set)} rules. Candidates: {rule_ids}")


                trace_lines.append(f"\nCONFLICT SET ({len(conflict_set)} rules applicable):")
                for rule in conflict_set:
                    premises_text = " AND ".join([f"{p.attribute}={p.value}" for p in rule.premises])
                    conclusion_text = f"{rule.conclusion.attribute}={rule.conclusion.value}"
                    trace_lines.append(f"  [Rule {rule.id}] IF ({premises_text}) THEN {conclusion_text}")


                self.logger.debug(f"[STEP {iterations}] Conflict Set Details:")
                for rule in conflict_set:

                    premises_text = " AND ".join([f"{p.attribute}={p.value}" for p in rule.premises])
                    conclusion_text = f"{rule.conclusion.attribute}={rule.conclusion.value}"
                    complexity = len(rule.premises)

                    debug_msg = f"  [CANDIDATE] Rule {rule.id}: IF {premises_text} THEN {conclusion_text} | Complexity: {complexity}"


                    if is_recency_strategy:

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


            if conflict_set:
                if greedy:

                    fired_count = 0
                    new_facts_this_iteration = []

                    for rule in conflict_set:

                        if rule.conclusion not in facts:
                            facts.add(rule.conclusion)
                            new_facts.append(rule.conclusion)
                            rules_fired.append(rule)
                            new_facts_this_iteration.append(rule.conclusion)
                            fired_count += 1


                            fired_rules_ids.add(rule.id)


                            if is_recency_strategy:
                                facts_with_recency[rule.conclusion] = iterations


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

                    if is_recency_strategy:

                        selected_rule = self.strategy.select(conflict_set, facts_with_recency)
                    else:

                        selected_rule = self.strategy.select(conflict_set, facts)

                    self.logger.info(f"[STRATEGY] {self.strategy.__class__.__name__} selected Rule {selected_rule.id}: {selected_rule}")


                    selected_premises = " AND ".join([f"{p.attribute}={p.value}" for p in selected_rule.premises])
                    selected_conclusion = f"{selected_rule.conclusion.attribute}={selected_rule.conclusion.value}"
                    trace_lines.append(f"\nSTRATEGY DECISION:")
                    trace_lines.append(f"  Strategy: {self.strategy.__class__.__name__}")
                    trace_lines.append(f"  Selected Rule: [Rule {selected_rule.id}] IF ({selected_premises}) THEN {selected_conclusion}")


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


                    if selected_rule.conclusion not in facts:
                        new_facts_from_rule = [selected_rule.conclusion]


                        facts.add(selected_rule.conclusion)
                        new_facts.append(selected_rule.conclusion)
                        rules_fired.append(selected_rule)


                        fired_rules_ids.add(selected_rule.id)

                        self.logger.info(f"[FIRE] Rule {selected_rule.id} fired! New facts inferred: {new_facts_from_rule}")


                        trace_lines.append(f"\nACTION:")
                        trace_lines.append(f"  Rule Fired!")
                        trace_lines.append(f"  New Fact Added: {selected_rule.conclusion.attribute}={selected_rule.conclusion.value}")


                        if is_recency_strategy:
                            facts_with_recency[selected_rule.conclusion] = iterations

                        inferred = True
                    else:
                        self.logger.info(f"[SKIP] Rule {selected_rule.id} activated but provided NO new facts.")
                        trace_lines.append(f"\nACTION:")
                        trace_lines.append(f"  Rule SKIPPED (conclusion already known)")


                    trace.append("\n".join(trace_lines))



                if _goal_achieved(facts, goal):
                    end_time = time.perf_counter()
                    execution_time_ms = (end_time - start_time) * 1000

                    self.logger.info(f"=== Goal {goal} ACHIEVED in iteration {iterations} ===")
                    self.logger.info(f"Execution time: {execution_time_ms:.3f} ms")


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


        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000



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












    def run(self, kb: KnowledgeBase, goal: Optional[Union[Fact, str]] = None) -> InferenceResult:















        start_time = time.perf_counter()


        facts = kb.facts.copy()
        new_facts = []
        rules_fired = []
        iterations = 0


        rules_evaluated = 0
        rules_activated = 0


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


            conflict_set = []
            for rule in kb.rules:
                rules_evaluated += 1
                if rule.is_satisfied_by(facts) and rule.conclusion not in facts:
                    conflict_set.append(rule)

            rules_activated += len(conflict_set)

            default_logger.debug(f"Conflict Set size: {len(conflict_set)}, Rule IDs: {[r.id for r in conflict_set]}")


            if conflict_set:
                for rule in conflict_set:
                    facts.add(rule.conclusion)
                    new_facts.append(rule.conclusion)
                    rules_fired.append(rule)
                    default_logger.info(f"Fired Rule {rule.id}, New Fact: {rule.conclusion}")

                inferred = True



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


        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000



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



















    def __init__(self, strategy: ConflictResolutionStrategy, run_id: Optional[str] = None, logger: Optional[logging.Logger] = None):








        self.strategy = strategy
        self.run_id = run_id
        self.logger = logger if logger else default_logger

    def run(self, kb: KnowledgeBase, goal: Fact) -> InferenceResult:


















        start_time = time.perf_counter()


        facts = kb.facts.copy()
        new_facts = []
        rules_fired = []


        proof_path = set()


        metrics = {
            'rules_evaluated': 0,
            'rules_activated': 0,
            'recursion_depth': 0,
            'max_depth': 0
        }

        self.logger.info(f"=== Starting Backward Chaining Inference {f'(Run ID: {self.run_id})' if self.run_id else ''} ===")


        self.logger.debug(f"Inference started. Knowledge Base contains {len(kb.rules)} rules and {len(facts)} initial facts.")

        self.logger.info(f"Initial facts: {len(facts)}, Rules: {len(kb.rules)}, Goal: {goal}")


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
            iterations=len(rules_fired),
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

















        metrics['max_depth'] = max(metrics['max_depth'], depth)
        indent = "  " * depth


        if goal in facts:
            self.logger.debug(f"{indent}[DEPTH {depth}] Goal {goal} already in facts (base case)")
            return True


        if goal in proof_path:
            self.logger.debug(f"{indent}[DEPTH {depth}] CYCLE DETECTED for goal {goal} - stopping recursion")
            self.logger.info(f"[BACKWARD] Cycle detected for {goal}, backtracking...")
            return False


        proof_path.add(goal)

        self.logger.info(f"[BACKWARD] Trying to prove goal: {goal}")
        self.logger.debug(f"{indent}[DEPTH {depth}] Attempting to prove: {goal}")


        competitive_rules = []
        for rule in rules:
            metrics['rules_evaluated'] += 1
            if rule.conclusion == goal:
                competitive_rules.append(rule)

        if not competitive_rules:

            self.logger.debug(f"{indent}[DEPTH {depth}] No competitive rules for goal {goal} - FAIL")
            self.logger.info(f"[BACKWARD] No rules can prove {goal}")
            proof_path.remove(goal)
            return False

        metrics['rules_activated'] += len(competitive_rules)


        rule_ids = [r.id for r in competitive_rules]
        self.logger.info(f"[BACKWARD] Found {len(competitive_rules)} competitive rules for {goal}: {rule_ids}")


        self.logger.debug(f"{indent}[DEPTH {depth}] Competitive Rules Details:")
        for rule in competitive_rules:
            premises_text = " AND ".join([f"{p.attribute}={p.value}" for p in rule.premises])
            conclusion_text = f"{rule.conclusion.attribute}={rule.conclusion.value}"
            complexity = len(rule.premises)
            self.logger.debug(f"{indent}  [CANDIDATE] Rule {rule.id}: IF {premises_text} THEN {conclusion_text} | Complexity: {complexity}")


        ordered_rules = self._order_rules_by_strategy(competitive_rules, facts)
        self.logger.debug(f"{indent}[DEPTH {depth}] Strategy ordered rules: {[r.id for r in ordered_rules]}")


        for rule in ordered_rules:
            self.logger.debug(f"{indent}[DEPTH {depth}] Trying rule {rule.id} with {len(rule.premises)} premises")


            all_premises_proven = True

            for i, premise in enumerate(rule.premises, 1):
                self.logger.debug(f"{indent}[DEPTH {depth}] Proving premise {i}/{len(rule.premises)}: {premise}")


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

                    self.logger.debug(f"{indent}[DEPTH {depth}] Premise {premise} FAILED - cannot prove")
                    all_premises_proven = False
                    break


            if all_premises_proven:

                facts.add(goal)
                new_facts.append(goal)
                rules_fired.append(rule)

                self.logger.info(f"[SUCCESS] Proved {goal} using Rule {rule.id}")
                self.logger.debug(f"{indent}[DEPTH {depth}] SUCCESS - All premises proven for rule {rule.id}")


                proof_path.remove(goal)
                return True


            self.logger.debug(f"{indent}[DEPTH {depth}] Rule {rule.id} failed, backtracking to next candidate...")


        self.logger.debug(f"{indent}[DEPTH {depth}] All {len(competitive_rules)} rules exhausted for {goal} - FAIL")
        self.logger.info(f"[BACKWARD] All rules failed for {goal}, cannot prove")
        proof_path.remove(goal)
        return False

    def _order_rules_by_strategy(
        self,
        rules: List[Rule],
        facts: Set[Fact]
    ) -> List[Rule]:













        if len(rules) == 1:
            return rules

        ordered = []
        remaining = rules.copy()

        while remaining:

            selected = self.strategy.select(remaining, facts)
            ordered.append(selected)
            remaining.remove(selected)

        return ordered