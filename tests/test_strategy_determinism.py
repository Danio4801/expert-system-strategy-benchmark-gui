












import pytest
from core.models import Fact, Rule, KnowledgeBase
from core.strategies import RandomStrategy, FirstStrategy, SpecificityStrategy, RecencyStrategy
from core.inference import ForwardChaining


def test_random_strategy_with_seed_is_deterministic():






    rules = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("A", "1")], Fact("C", "1")),
        Rule(3, [Fact("A", "1")], Fact("D", "1")),
        Rule(4, [Fact("A", "1")], Fact("E", "1")),
        Rule(5, [Fact("A", "1")], Fact("F", "1"))
    ]
    facts = {Fact("A", "1")}


    kb1 = KnowledgeBase(rules=rules, facts=facts.copy())
    engine1 = ForwardChaining(RandomStrategy(seed=42))
    result1 = engine1.run(kb1)


    kb2 = KnowledgeBase(rules=rules, facts=facts.copy())
    engine2 = ForwardChaining(RandomStrategy(seed=42))
    result2 = engine2.run(kb2)


    assert result1.rules_fired[0].id == result2.rules_fired[0].id, \
        "RandomStrategy with same seed must select the same rule"


    fired_ids_1 = [r.id for r in result1.rules_fired]
    fired_ids_2 = [r.id for r in result2.rules_fired]
    assert fired_ids_1 == fired_ids_2, \
        f"Expected identical rule sequences, got {fired_ids_1} vs {fired_ids_2}"


def test_random_strategy_with_different_seeds():





    rules = [
        Rule(i, [Fact("A", "1")], Fact(f"Out{i}", "1"))
        for i in range(1, 11)
    ]
    facts = {Fact("A", "1")}


    kb1 = KnowledgeBase(rules=rules, facts=facts.copy())
    engine1 = ForwardChaining(RandomStrategy(seed=42))
    result1 = engine1.run(kb1)


    kb2 = KnowledgeBase(rules=rules, facts=facts.copy())
    engine2 = ForwardChaining(RandomStrategy(seed=999))
    result2 = engine2.run(kb2)


    fired_ids_1 = [r.id for r in result1.rules_fired]
    fired_ids_2 = [r.id for r in result2.rules_fired]




    assert len(fired_ids_1) > 0
    assert len(fired_ids_2) > 0


def test_random_strategy_select_directly():



    conflict_set = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("A", "1")], Fact("C", "1")),
        Rule(3, [Fact("A", "1")], Fact("D", "1"))
    ]
    facts = {Fact("A", "1")}


    strategy1 = RandomStrategy(seed=42)
    selected1 = strategy1.select(conflict_set, facts)


    strategy2 = RandomStrategy(seed=42)
    selected2 = strategy2.select(conflict_set, facts)


    assert selected1.id == selected2.id, "Same seed must produce same selection"


def test_random_strategy_multiple_selections_with_seed():






    conflict_set = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("A", "1")], Fact("C", "1")),
        Rule(3, [Fact("A", "1")], Fact("D", "1"))
    ]
    facts = {Fact("A", "1")}


    strategy1 = RandomStrategy(seed=42)
    seq1 = [strategy1.select(conflict_set, facts).id for _ in range(5)]


    strategy2 = RandomStrategy(seed=42)
    seq2 = [strategy2.select(conflict_set, facts).id for _ in range(5)]


    assert seq1 == seq2, f"Expected identical sequences, got {seq1} vs {seq2}"


def test_first_strategy_always_deterministic():



    conflict_set = [
        Rule(5, [Fact("A", "1")], Fact("X", "1")),
        Rule(2, [Fact("A", "1")], Fact("Y", "1")),
        Rule(7, [Fact("A", "1")], Fact("Z", "1"))
    ]
    facts = {Fact("A", "1")}

    strategy = FirstStrategy()


    selections = [strategy.select(conflict_set, facts).id for _ in range(10)]


    assert all(s == 5 for s in selections), "FirstStrategy must always select first rule"


def test_specificity_strategy_deterministic():





    conflict_set = [
        Rule(1, [Fact("A", "1")], Fact("X", "1")),
        Rule(2, [Fact("A", "1"), Fact("B", "1")], Fact("Y", "1")),
        Rule(3, [Fact("A", "1")], Fact("Z", "1"))
    ]
    facts = {Fact("A", "1"), Fact("B", "1")}

    strategy = SpecificityStrategy()


    selections = [strategy.select(conflict_set, facts).id for _ in range(10)]


    assert all(s == 2 for s in selections), "SpecificityStrategy must always select longest rule"


def test_recency_strategy_deterministic():



    conflict_set = [
        Rule(1, [Fact("A", "1")], Fact("X", "1")),
        Rule(2, [Fact("B", "1")], Fact("Y", "1"))
    ]
    facts_with_recency = {
        Fact("A", "1"): 0,
        Fact("B", "1"): 5
    }

    strategy = RecencyStrategy()


    selections = [strategy.select(conflict_set, facts_with_recency).id for _ in range(10)]


    assert all(s == 2 for s in selections), "RecencyStrategy must always select rule with newest fact"


def test_random_strategy_without_seed_works():





    conflict_set = [
        Rule(1, [Fact("A", "1")], Fact("B", "1")),
        Rule(2, [Fact("A", "1")], Fact("C", "1"))
    ]
    facts = {Fact("A", "1")}

    strategy = RandomStrategy()


    selections = [strategy.select(conflict_set, facts) for _ in range(10)]


    assert all(s in conflict_set for s in selections)


def test_inference_with_seeded_random_is_reproducible():





    rules = [
        Rule(i, [Fact("A", "1")], Fact(f"Out{i}", "1"))
        for i in range(1, 21)
    ]
    facts = {Fact("A", "1")}


    kb1 = KnowledgeBase(rules=rules, facts=facts.copy())
    engine1 = ForwardChaining(RandomStrategy(seed=12345))
    result1 = engine1.run(kb1)


    kb2 = KnowledgeBase(rules=rules, facts=facts.copy())
    engine2 = ForwardChaining(RandomStrategy(seed=12345))
    result2 = engine2.run(kb2)


    assert result1.iterations == result2.iterations
    assert result1.rules_evaluated == result2.rules_evaluated
    assert result1.rules_activated == result2.rules_activated
    assert result1.facts_count == result2.facts_count


    fired_ids_1 = [r.id for r in result1.rules_fired]
    fired_ids_2 = [r.id for r in result2.rules_fired]
    assert fired_ids_1 == fired_ids_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
