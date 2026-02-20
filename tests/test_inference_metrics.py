















import pytest
from core.models import Fact, Rule, KnowledgeBase
from core.strategies import FirstStrategy, RandomStrategy, SpecificityStrategy
from core.inference import ForwardChaining, GreedyForwardChaining, BackwardChaining


def test_forward_chaining_rules_evaluated():









    rules = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("B", "1")], Fact("C", "1")),
        Rule(3, [Fact("X", "1")], Fact("Y", "1"))
    ]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = ForwardChaining(FirstStrategy())


    result = engine.run(kb)





    assert result.iterations == 3
    assert result.rules_evaluated == 9
    assert len(result.rules_fired) == 2


def test_forward_chaining_rules_activated():








    rules = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("B", "1")], Fact("C", "1"))
    ]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = ForwardChaining(FirstStrategy())
    result = engine.run(kb)


    assert result.rules_activated == 2


def test_forward_chaining_facts_count():

    rules = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("B", "1")], Fact("C", "1"))
    ]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = ForwardChaining(FirstStrategy())
    result = engine.run(kb)





    assert result.facts_count == 3
    assert len(result.facts) == 3


def test_forward_chaining_execution_time_ms():

    rules = [Rule(1, [Fact("A", "1")], Fact("B", "1"))]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = ForwardChaining(FirstStrategy())
    result = engine.run(kb)


    assert hasattr(result, 'execution_time_ms')
    assert result.execution_time_ms > 0
    assert result.execution_time_ms < 1000


def test_greedy_forward_chaining_metrics():





    rules = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("A", "1")], Fact("C", "1")),
    ]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = GreedyForwardChaining()
    result = engine.run(kb)




    assert result.iterations == 2
    assert result.rules_evaluated == 4
    assert result.rules_activated == 2
    assert len(result.rules_fired) == 2
    assert result.facts_count == 3


def test_backward_chaining_metrics():









    rules = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("B", "1")], Fact("C", "1"))
    ]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = BackwardChaining(FirstStrategy())
    result = engine.run(kb, goal=Fact("C", "1"))


    assert result.success is True
    assert len(result.rules_fired) == 2
    assert result.rules_evaluated > 0
    assert result.rules_activated > 0
    assert result.facts_count == 3


def test_metrics_with_no_inference():







    rules = [
        Rule(1, [Fact("X", "1")], Fact("Y", "1"))
    ]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = ForwardChaining(FirstStrategy())
    result = engine.run(kb)



    assert result.iterations == 1
    assert result.rules_evaluated == 1
    assert result.rules_activated == 0
    assert len(result.rules_fired) == 0
    assert result.facts_count == 1


def test_metrics_multiple_conflict_sets():








    rules = [

        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("A", "1")], Fact("C", "1")),
        Rule(3, [Fact("A", "1")], Fact("D", "1")),

        Rule(4, [Fact("B", "1")], Fact("E", "1"))
    ]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = ForwardChaining(FirstStrategy())
    result = engine.run(kb)







    assert result.rules_activated == 9
    assert len(result.rules_fired) == 4


def test_execution_time_is_positive():

    rules = [Rule(1, [Fact("A", "1")], Fact("B", "1"))]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    for engine_class in [ForwardChaining, GreedyForwardChaining]:
        if engine_class == ForwardChaining:
            engine = engine_class(FirstStrategy())
        else:
            engine = engine_class()

        result = engine.run(kb)
        assert result.execution_time_ms > 0, f"{engine_class.__name__} execution_time_ms must be > 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
