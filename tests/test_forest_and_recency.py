




















import pytest
import pandas as pd
from core.models import Fact, Rule, KnowledgeBase
from core.strategies import RecencyStrategy
from core.inference import ForwardChaining
from preprocessing.forest_rule_generator import ForestRuleGenerator






def test_logical_clock_recency_basic():











    rule1 = Rule(1, [Fact("A", "1")], Fact("Out1", "1"))
    rule2 = Rule(2, [Fact("B", "1")], Fact("Out2", "1"))
    conflict_set = [rule1, rule2]


    facts_with_recency = {
        Fact("A", "1"): 0,
        Fact("B", "1"): 1
    }


    strategy = RecencyStrategy()
    selected = strategy.select(conflict_set, facts_with_recency)


    assert selected.id == 2, "RecencyStrategy should select rule using fact with higher clock_id"


def test_logical_clock_increments_during_inference():












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





    assert fired_ids[0] == 1, "First iteration should fire R1"
    assert fired_ids[1] == 3, "Second iteration: RecencyStrategy should pick R3 (uses newer fact B)"
    assert fired_ids[2] == 2, "Third iteration should fire R2"


def test_logical_clock_with_multiple_premises():









    rule1 = Rule(1, [Fact("A", "1"), Fact("B", "1")], Fact("Out1", "1"))
    rule2 = Rule(2, [Fact("C", "1"), Fact("D", "1")], Fact("Out2", "1"))
    conflict_set = [rule1, rule2]

    facts_with_recency = {
        Fact("A", "1"): 1,
        Fact("B", "1"): 5,
        Fact("C", "1"): 3,
        Fact("D", "1"): 4
    }


    strategy = RecencyStrategy()
    selected = strategy.select(conflict_set, facts_with_recency)


    assert selected.id == 1, "RecencyStrategy should select rule with highest max clock_id"






def test_variable_depth_forest_generates_rules_of_varying_complexity():










    df = pd.DataFrame({
        'feature1': ['low', 'low', 'high', 'high'] * 10,
        'feature2': ['young', 'old', 'young', 'old'] * 10,
        'feature3': ['yes', 'no', 'yes', 'no'] * 10,
        'decision': ['accept', 'reject', 'accept', 'reject'] * 10
    })


    generator = ForestRuleGenerator(
        n_estimators=5,
        min_depth=2,
        max_depth=5,
        random_state=42
    )
    rules = generator.generate(df, decision_column='decision')


    assert len(rules) > 0, "Should generate at least one rule"


    complexities = [len(rule.premises) for rule in rules]
    unique_complexities = set(complexities)


    assert len(unique_complexities) > 1, (
        f"Variable Depth Forest should produce rules of varying complexity. "
        f"Got only {unique_complexities} unique complexity values. "
        f"Expected at least 2 different values from range [2, 5]."
    )



    min_complexity = min(complexities)
    max_complexity = max(complexities)
    assert min_complexity >= 1, "Minimum rule complexity should be at least 1"
    assert max_complexity <= 5, "Maximum rule complexity should not exceed max_depth"


def test_variable_depth_forest_trees_have_different_depths():










    df = pd.DataFrame({
        'f1': ['a', 'b', 'c', 'd'] * 15,
        'f2': ['x', 'y', 'z', 'w'] * 15,
        'f3': ['1', '2', '1', '2'] * 15,
        'target': ['pos', 'neg', 'pos', 'neg'] * 15
    })


    generator = ForestRuleGenerator(
        n_estimators=10,
        min_depth=2,
        max_depth=5,
        random_state=123
    )
    rules = generator.generate(df, decision_column='target')


    tree_depths = [tree.get_depth() for tree in generator.estimators_]
    unique_depths = set(tree_depths)

    assert len(unique_depths) > 1, (
        f"Trees should have varying depths. Got only {unique_depths}. "
        f"Expected at least 2 different depth values from range [{generator.min_depth}, {generator.max_depth}]."
    )






def test_seeding_produces_identical_rules_with_same_seed():








    df = pd.DataFrame({
        'attr1': ['a', 'b', 'a', 'b'] * 12,
        'attr2': ['x', 'y', 'x', 'y'] * 12,
        'class': ['yes', 'no', 'yes', 'no'] * 12
    })


    gen1 = ForestRuleGenerator(n_estimators=5, min_depth=2, max_depth=4, random_state=999)
    rules1 = gen1.generate(df, decision_column='class')


    gen2 = ForestRuleGenerator(n_estimators=5, min_depth=2, max_depth=4, random_state=999)
    rules2 = gen2.generate(df, decision_column='class')


    assert len(rules1) == len(rules2), (
        f"Same seed should produce same number of rules. "
        f"Got {len(rules1)} vs {len(rules2)}"
    )


    for i, (r1, r2) in enumerate(zip(rules1, rules2)):

        assert len(r1.premises) == len(r2.premises), (
            f"Rule {i}: Different number of premises ({len(r1.premises)} vs {len(r2.premises)})"
        )


        premises1 = set(r1.premises)
        premises2 = set(r2.premises)
        assert premises1 == premises2, (
            f"Rule {i}: Different premises. Got {premises1} vs {premises2}"
        )


        assert r1.conclusion == r2.conclusion, (
            f"Rule {i}: Different conclusions. Got {r1.conclusion} vs {r2.conclusion}"
        )


def test_seeding_produces_different_rules_with_different_seed():









    df = pd.DataFrame({
        'feat1': ['val1', 'val2', 'val3', 'val4'] * 15,
        'feat2': ['a', 'b', 'c', 'd'] * 15,
        'target': ['good', 'bad', 'good', 'bad'] * 15
    })


    gen1 = ForestRuleGenerator(n_estimators=7, min_depth=2, max_depth=5, random_state=999)
    rules1 = gen1.generate(df, decision_column='target')


    gen2 = ForestRuleGenerator(n_estimators=7, min_depth=2, max_depth=5, random_state=123)
    rules2 = gen2.generate(df, decision_column='target')



    depths1 = [tree.get_depth() for tree in gen1.estimators_]
    depths2 = [tree.get_depth() for tree in gen2.estimators_]


    assert depths1 != depths2, (
        f"Different seeds should produce different tree depths. "
        f"Got identical depths: {depths1}"
    )



    if len(rules1) == len(rules2):
        rules_are_different = False
        for r1, r2 in zip(rules1, rules2):
            if set(r1.premises) != set(r2.premises) or r1.conclusion != r2.conclusion:
                rules_are_different = True
                break

        assert rules_are_different, (
            "Different seeds should produce at least some different rules"
        )






def test_logical_clock_with_initial_facts_have_clock_zero():






    rule1 = Rule(1, [Fact("A", "1")], Fact("Out1", "1"))
    rule2 = Rule(2, [Fact("B", "1")], Fact("Out2", "1"))
    rule3 = Rule(3, [Fact("C", "1")], Fact("Out3", "1"))
    conflict_set = [rule1, rule2, rule3]

    facts_with_recency = {
        Fact("A", "1"): 0,
        Fact("B", "1"): 0,
        Fact("C", "1"): 0
    }


    strategy = RecencyStrategy()
    selected = strategy.select(conflict_set, facts_with_recency)


    assert selected.id == 1, "With all facts having clock_id=0, should select first rule"


def test_variable_depth_with_min_equals_max():






    df = pd.DataFrame({
        'x': ['a', 'b'] * 20,
        'y': ['1', '2'] * 20,
        'z': ['yes', 'no'] * 20
    })


    generator = ForestRuleGenerator(
        n_estimators=5,
        min_depth=3,
        max_depth=3,
        random_state=555
    )
    rules = generator.generate(df, decision_column='z')


    for tree in generator.estimators_:

        assert tree.get_depth() <= 3, "Tree depth should not exceed max_depth"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
