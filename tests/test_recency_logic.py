

















import pytest
from core.models import Fact, Rule, KnowledgeBase
from core.strategies import RecencyStrategy
from core.inference import ForwardChaining


def test_recency_strategy_selects_rule_with_newest_fact():










    conflict_set = [
        Rule(1, [Fact("A", "1")], Fact("Out1", "1")),
        Rule(2, [Fact("B", "1")], Fact("Out2", "1"))
    ]
    facts_with_recency = {
        Fact("A", "1"): 1,
        Fact("B", "1"): 5
    }

    strategy = RecencyStrategy()
    selected = strategy.select(conflict_set, facts_with_recency)


    assert selected.id == 2, "RecencyStrategy should select rule using newest fact (B)"


def test_recency_strategy_with_multiple_premises():








    conflict_set = [
        Rule(1, [Fact("A", "1"), Fact("B", "1")], Fact("Out1", "1")),
        Rule(2, [Fact("C", "1"), Fact("D", "1")], Fact("Out2", "1"))
    ]
    facts_with_recency = {
        Fact("A", "1"): 1,
        Fact("B", "1"): 10,
        Fact("C", "1"): 5,
        Fact("D", "1"): 7
    }

    strategy = RecencyStrategy()
    selected = strategy.select(conflict_set, facts_with_recency)


    assert selected.id == 1, "RecencyStrategy should select rule with highest max recency"


def test_recency_strategy_with_tie():








    conflict_set = [
        Rule(1, [Fact("A", "1")], Fact("Out1", "1")),
        Rule(2, [Fact("B", "1")], Fact("Out2", "1"))
    ]
    facts_with_recency = {
        Fact("A", "1"): 5,
        Fact("B", "1"): 5
    }

    strategy = RecencyStrategy()
    selected = strategy.select(conflict_set, facts_with_recency)


    assert selected.id == 1


def test_recency_strategy_in_forward_chaining():











    rules = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("A", "1")], Fact("C", "1")),
        Rule(3, [Fact("B", "1")], Fact("D", "1"))
    ]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = ForwardChaining(RecencyStrategy())
    result = engine.run(kb)





    fired_ids = [r.id for r in result.rules_fired]
    assert fired_ids[0] == 1
    assert fired_ids[1] == 3
    assert fired_ids[2] == 2


def test_recency_increments_with_iterations():





    rules = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("B", "1")], Fact("C", "1")),
        Rule(3, [Fact("C", "1")], Fact("D", "1"))
    ]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = ForwardChaining(RecencyStrategy())
    result = engine.run(kb)







    assert result.iterations == 4
    assert len(result.new_facts) == 3


def test_recency_with_initial_facts_have_recency_zero():








    rules = [
        Rule(1, [Fact("A", "1")], Fact("Out1", "1")),
        Rule(2, [Fact("B", "1")], Fact("Out2", "1")),
        Rule(3, [Fact("C", "1")], Fact("Out3", "1"))
    ]
    facts = {Fact("A", "1"), Fact("B", "1"), Fact("C", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = ForwardChaining(RecencyStrategy())
    result = engine.run(kb)



    fired_ids = [r.id for r in result.rules_fired]
    assert fired_ids[0] == 1


def test_recency_complex_scenario():





    rules = [

        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("A", "1")], Fact("C", "1")),

        Rule(3, [Fact("B", "1")], Fact("D", "1")),
        Rule(4, [Fact("C", "1")], Fact("E", "1")),

        Rule(5, [Fact("A", "1"), Fact("D", "1")], Fact("F", "1")),
        Rule(6, [Fact("B", "1"), Fact("E", "1")], Fact("G", "1"))
    ]
    facts = {Fact("A", "1")}
    kb = KnowledgeBase(rules=rules, facts=facts)

    engine = ForwardChaining(RecencyStrategy())
    result = engine.run(kb)


    assert result.iterations >= 3
    assert len(result.rules_fired) >= 3



def test_recency_vs_first_strategy_difference():





    rules = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("A", "1")], Fact("C", "1")),
        Rule(3, [Fact("B", "1")], Fact("D", "1")),
        Rule(4, [Fact("C", "1")], Fact("E", "1"))
    ]
    facts = {Fact("A", "1")}


    kb1 = KnowledgeBase(rules=rules, facts=facts.copy())
    from core.strategies import FirstStrategy
    engine1 = ForwardChaining(FirstStrategy())
    result1 = engine1.run(kb1)


    kb2 = KnowledgeBase(rules=rules, facts=facts.copy())
    engine2 = ForwardChaining(RecencyStrategy())
    result2 = engine2.run(kb2)


    fired_ids_first = [r.id for r in result1.rules_fired]
    fired_ids_recency = [r.id for r in result2.rules_fired]




    assert len(fired_ids_first) >= 2
    assert len(fired_ids_recency) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
