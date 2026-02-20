





















import pytest
from core.models import Fact, Rule, KnowledgeBase
from core.strategies import FirstStrategy, RandomStrategy, SpecificityStrategy, RecencyStrategy
from core.inference import ForwardChaining, GreedyForwardChaining, BackwardChaining, InferenceResult


class TestInferenceResult:


    
    def test_create_inference_result(self):

        result = InferenceResult(
            success=True,
            facts={Fact("a", "1")},
            new_facts=[Fact("a", "1")],
            rules_fired=[],
            iterations=1,
            execution_time_ms=1.0,
            rules_evaluated=10,
            rules_activated=5,
            facts_count=1
        )
        assert result.success == True
        assert result.iterations == 1
        assert result.execution_time_ms == 1.0
        assert result.rules_evaluated == 10
        assert result.rules_activated == 5
        assert result.facts_count == 1


class TestForwardChainingBasic:


    
    def test_empty_kb_returns_no_new_facts(self):

        kb = KnowledgeBase(rules=[], facts={Fact("a", "1")})
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        
        result = engine.run(kb)
        
        assert len(result.new_facts) == 0
        assert result.iterations == 1
    
    def test_single_rule_fires_when_premises_satisfied(self):

        rule = Rule(
            id=1,
            premises=[Fact("a", "1")],
            conclusion=Fact("b", "2")
        )
        kb = KnowledgeBase(
            rules=[rule],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        
        result = engine.run(kb)
        
        assert Fact("b", "2") in result.facts
        assert Fact("b", "2") in result.new_facts
        assert rule in result.rules_fired
    
    def test_rule_does_not_fire_when_premises_not_satisfied(self):

        rule = Rule(
            id=1,
            premises=[Fact("a", "1"), Fact("c", "3")],
            conclusion=Fact("b", "2")
        )
        kb = KnowledgeBase(
            rules=[rule],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        
        result = engine.run(kb)
        
        assert Fact("b", "2") not in result.facts
        assert len(result.new_facts) == 0
    
    def test_rule_does_not_fire_twice(self):


        rule = Rule(
            id=1,
            premises=[Fact("a", "1")],
            conclusion=Fact("b", "2")
        )
        kb = KnowledgeBase(
            rules=[rule],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        
        result = engine.run(kb)
        

        assert result.rules_fired.count(rule) == 1


class TestForwardChainingChain:


    
    def test_chain_of_two_rules(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(id=2, premises=[Fact("b", "2")], conclusion=Fact("c", "3"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        
        result = engine.run(kb)
        
        assert Fact("b", "2") in result.facts
        assert Fact("c", "3") in result.facts
        assert len(result.new_facts) == 2
        assert result.iterations == 3
    
    def test_chain_of_three_rules(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(id=2, premises=[Fact("b", "2")], conclusion=Fact("c", "3"))
        rule3 = Rule(id=3, premises=[Fact("c", "3")], conclusion=Fact("d", "4"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2, rule3],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        
        result = engine.run(kb)
        
        assert Fact("d", "4") in result.facts
        assert len(result.new_facts) == 3


class TestForwardChainingWithGoal:


    
    def test_stops_when_goal_reached(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(id=2, premises=[Fact("b", "2")], conclusion=Fact("c", "3"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        

        result = engine.run(kb, goal=Fact("b", "2"))
        
        assert result.success == True
        assert Fact("b", "2") in result.facts
        assert Fact("c", "3") not in result.facts
    
    def test_returns_failure_when_goal_not_reachable(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        
        kb = KnowledgeBase(
            rules=[rule1],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        

        result = engine.run(kb, goal=Fact("x", "y"))
        
        assert result.success == False
    
    def test_success_true_when_no_goal(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        
        kb = KnowledgeBase(
            rules=[rule1],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        
        result = engine.run(kb)
        
        assert result.success == True


class TestForwardChainingConflictSet:


    
    def test_specificity_strategy_selects_longer_rule(self):


        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("x", "1"))
        rule2 = Rule(
            id=2,
            premises=[Fact("a", "1"), Fact("b", "2")],
            conclusion=Fact("x", "2")
        )
        
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1"), Fact("b", "2")}
        )
        strategy = SpecificityStrategy()
        engine = ForwardChaining(strategy)
        
        result = engine.run(kb)
        

        assert result.rules_fired[0].id == 2
    
    def test_first_strategy_selects_first_rule(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("x", "1"))
        rule2 = Rule(id=2, premises=[Fact("a", "1")], conclusion=Fact("x", "2"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        
        result = engine.run(kb)
        
        assert result.rules_fired[0].id == 1


class TestForwardChainingMetrics:


    
    def test_execution_time_is_recorded(self):

        rule = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        kb = KnowledgeBase(rules=[rule], facts={Fact("a", "1")})
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)

        result = engine.run(kb)

        assert result.execution_time_ms >= 0
        assert isinstance(result.execution_time_ms, float)
    
    def test_iterations_count_is_correct(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(id=2, premises=[Fact("b", "2")], conclusion=Fact("c", "3"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = ForwardChaining(strategy)
        
        result = engine.run(kb)
        

        assert result.iterations == 3


class TestGreedyForwardChaining:


    
    def test_fires_all_applicable_rules_at_once(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("x", "1"))
        rule2 = Rule(id=2, premises=[Fact("a", "1")], conclusion=Fact("y", "2"))
        rule3 = Rule(id=3, premises=[Fact("a", "1")], conclusion=Fact("z", "3"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2, rule3],
            facts={Fact("a", "1")}
        )
        engine = GreedyForwardChaining()
        
        result = engine.run(kb)
        

        assert Fact("x", "1") in result.facts
        assert Fact("y", "2") in result.facts
        assert Fact("z", "3") in result.facts
        assert result.iterations == 2
    
    def test_greedy_completes_faster(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("x", "1"))
        rule2 = Rule(id=2, premises=[Fact("a", "1")], conclusion=Fact("y", "2"))
        
        kb_greedy = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1")}
        )
        kb_normal = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1")}
        )
        
        greedy_engine = GreedyForwardChaining()
        normal_engine = ForwardChaining(FirstStrategy())
        
        greedy_result = greedy_engine.run(kb_greedy)
        normal_result = normal_engine.run(kb_normal)
        

        assert greedy_result.iterations < normal_result.iterations





class TestBackwardChainingBasic:


    
    def test_goal_already_in_facts(self):

        kb = KnowledgeBase(
            rules=[],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("a", "1"))
        
        assert result.success == True
        assert len(result.new_facts) == 0
    
    def test_goal_proven_by_single_rule(self):

        rule = Rule(
            id=1,
            premises=[Fact("a", "1")],
            conclusion=Fact("b", "2")
        )
        kb = KnowledgeBase(
            rules=[rule],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("b", "2"))
        
        assert result.success == True
        assert Fact("b", "2") in result.facts
        assert rule in result.rules_fired
    
    def test_goal_not_provable(self):

        rule = Rule(
            id=1,
            premises=[Fact("a", "1")],
            conclusion=Fact("b", "2")
        )
        kb = KnowledgeBase(
            rules=[rule],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        

        result = engine.run(kb, goal=Fact("x", "y"))
        
        assert result.success == False
    
    def test_goal_not_provable_missing_premise(self):

        rule = Rule(
            id=1,
            premises=[Fact("a", "1"), Fact("c", "3")],
            conclusion=Fact("b", "2")
        )
        kb = KnowledgeBase(
            rules=[rule],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("b", "2"))
        
        assert result.success == False


class TestBackwardChainingRecursion:


    
    def test_chain_of_two_rules(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(id=2, premises=[Fact("b", "2")], conclusion=Fact("c", "3"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("c", "3"))
        
        assert result.success == True
        assert Fact("b", "2") in result.facts
        assert Fact("c", "3") in result.facts
    
    def test_chain_of_three_rules(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(id=2, premises=[Fact("b", "2")], conclusion=Fact("c", "3"))
        rule3 = Rule(id=3, premises=[Fact("c", "3")], conclusion=Fact("d", "4"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2, rule3],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("d", "4"))
        
        assert result.success == True
        assert Fact("d", "4") in result.facts
    
    def test_multiple_premises_all_must_be_proven(self):




        rule = Rule(
            id=1,
            premises=[Fact("a", "1"), Fact("b", "2"), Fact("c", "3")],
            conclusion=Fact("d", "4")
        )
        kb = KnowledgeBase(
            rules=[rule],
            facts={Fact("a", "1"), Fact("b", "2"), Fact("c", "3")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("d", "4"))
        
        assert result.success == True
    
    def test_multiple_premises_one_provable_recursively(self):




        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(
            id=2,
            premises=[Fact("b", "2"), Fact("x", "9")],
            conclusion=Fact("c", "3")
        )
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1"), Fact("x", "9")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("c", "3"))
        
        assert result.success == True
        assert Fact("b", "2") in result.facts


class TestBackwardChainingBacktracking:


    
    def test_backtrack_to_second_rule(self):




        rule1 = Rule(id=1, premises=[Fact("p", "1")], conclusion=Fact("x", "9"))
        rule2 = Rule(id=2, premises=[Fact("q", "2")], conclusion=Fact("x", "9"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("q", "2")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("x", "9"))
        
        assert result.success == True

        assert rule2 in result.rules_fired
    
    def test_backtrack_deep_chain(self):




        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(id=2, premises=[Fact("b", "2")], conclusion=Fact("z", "9"))
        rule3 = Rule(id=3, premises=[Fact("c", "3")], conclusion=Fact("z", "9"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2, rule3],
            facts={Fact("c", "3")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("z", "9"))
        
        assert result.success == True


class TestBackwardChainingCycleDetection:


    
    def test_no_infinite_loop_on_cycle(self):



        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(id=2, premises=[Fact("b", "2")], conclusion=Fact("a", "1"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        

        result = engine.run(kb, goal=Fact("c", "3"))
        
        assert result.success == False


class TestBackwardChainingMetrics:


    
    def test_execution_time_recorded(self):

        kb = KnowledgeBase(rules=[], facts={Fact("a", "1")})
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)

        result = engine.run(kb, goal=Fact("a", "1"))

        assert result.execution_time_ms >= 0
        assert isinstance(result.execution_time_ms, float)
    
    def test_rules_fired_recorded(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(id=2, premises=[Fact("b", "2")], conclusion=Fact("c", "3"))
        
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1")}
        )
        strategy = FirstStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("c", "3"))
        
        assert len(result.rules_fired) == 2
        assert rule1 in result.rules_fired
        assert rule2 in result.rules_fired


class TestBackwardChainingStrategy:


    
    def test_specificity_strategy_chooses_longer_rule(self):




        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("x", "9"))
        rule2 = Rule(
            id=2,
            premises=[Fact("a", "1"), Fact("b", "2")],
            conclusion=Fact("x", "9")
        )
        
        kb = KnowledgeBase(
            rules=[rule1, rule2],
            facts={Fact("a", "1"), Fact("b", "2")}
        )
        strategy = SpecificityStrategy()
        engine = BackwardChaining(strategy)
        
        result = engine.run(kb, goal=Fact("x", "9"))
        
        assert result.success == True

        assert result.rules_fired[0].id == 2
