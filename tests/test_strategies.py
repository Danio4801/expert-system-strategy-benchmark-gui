















import pytest
from core.models import Fact, Rule
from core.strategies import (
    ConflictResolutionStrategy,
    RandomStrategy,
    FirstStrategy,
    RecencyStrategy,
    SpecificityStrategy
)


class TestRandomStrategy:


    
    def test_select_returns_rule_from_conflict_set(self):

        rules = [
            Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2")),
            Rule(id=2, premises=[Fact("c", "3")], conclusion=Fact("d", "4")),
        ]
        facts = {Fact("a", "1")}
        
        strategy = RandomStrategy()
        selected = strategy.select(rules, facts)
        
        assert selected in rules
    
    def test_select_with_single_rule(self):

        rule = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        facts = {Fact("a", "1")}
        
        strategy = RandomStrategy()
        selected = strategy.select([rule], facts)
        
        assert selected == rule
    
    def test_select_is_random(self):

        rules = [
            Rule(id=i, premises=[Fact("a", "1")], conclusion=Fact("x", str(i)))
            for i in range(10)
        ]
        facts = {Fact("a", "1")}
        
        strategy = RandomStrategy()
        

        selections = [strategy.select(rules, facts).id for _ in range(50)]
        unique_selections = set(selections)
        

        assert len(unique_selections) > 1


class TestFirstStrategy:


    
    def test_select_returns_first_rule(self):

        rules = [
            Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2")),
            Rule(id=2, premises=[Fact("c", "3")], conclusion=Fact("d", "4")),
            Rule(id=3, premises=[Fact("e", "5")], conclusion=Fact("f", "6")),
        ]
        facts = set()
        
        strategy = FirstStrategy()
        selected = strategy.select(rules, facts)
        
        assert selected.id == 1
    
    def test_select_with_single_rule(self):

        rule = Rule(id=5, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        facts = set()
        
        strategy = FirstStrategy()
        selected = strategy.select([rule], facts)
        
        assert selected.id == 5


class TestSpecificityStrategy:


    
    def test_select_returns_rule_with_most_premises(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("x", "1"))
        rule2 = Rule(id=2, premises=[Fact("a", "1"), Fact("b", "2"), Fact("c", "3")], conclusion=Fact("x", "2"))
        rule3 = Rule(id=3, premises=[Fact("a", "1"), Fact("b", "2")], conclusion=Fact("x", "3"))
        
        facts = set()
        strategy = SpecificityStrategy()
        selected = strategy.select([rule1, rule2, rule3], facts)
        
        assert selected.id == 2
    
    def test_select_with_equal_premises_returns_first(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1"), Fact("b", "2")], conclusion=Fact("x", "1"))
        rule2 = Rule(id=2, premises=[Fact("c", "3"), Fact("d", "4")], conclusion=Fact("x", "2"))
        
        facts = set()
        strategy = SpecificityStrategy()
        selected = strategy.select([rule1, rule2], facts)
        
        assert selected.id == 1
    
    def test_select_with_single_rule(self):

        rule = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        facts = set()
        
        strategy = SpecificityStrategy()
        selected = strategy.select([rule], facts)
        
        assert selected.id == 1


class TestRecencyStrategy:



    
    def test_select_returns_rule_with_most_recent_facts(self):

        old_fact = Fact("a", "1")
        new_fact = Fact("b", "2")
        
        rule1 = Rule(id=1, premises=[old_fact], conclusion=Fact("x", "1"))
        rule2 = Rule(id=2, premises=[new_fact], conclusion=Fact("x", "2"))
        


        facts_with_recency = {old_fact: 0, new_fact: 1}
        
        strategy = RecencyStrategy()
        selected = strategy.select([rule1, rule2], facts_with_recency)
        
        assert selected.id == 2
    
    def test_select_uses_max_recency_among_premises(self):

        fact_old = Fact("a", "1")
        fact_medium = Fact("b", "2")
        fact_new = Fact("c", "3")
        

        rule1 = Rule(id=1, premises=[fact_old, fact_medium], conclusion=Fact("x", "1"))

        rule2 = Rule(id=2, premises=[fact_new], conclusion=Fact("x", "2"))
        
        facts_with_recency = {fact_old: 0, fact_medium: 1, fact_new: 2}
        
        strategy = RecencyStrategy()
        selected = strategy.select([rule1, rule2], facts_with_recency)
        
        assert selected.id == 2
    
    def test_select_with_equal_recency_returns_first(self):

        fact1 = Fact("a", "1")
        fact2 = Fact("b", "2")
        
        rule1 = Rule(id=1, premises=[fact1], conclusion=Fact("x", "1"))
        rule2 = Rule(id=2, premises=[fact2], conclusion=Fact("x", "2"))
        

        facts_with_recency = {fact1: 1, fact2: 1}
        
        strategy = RecencyStrategy()
        selected = strategy.select([rule1, rule2], facts_with_recency)
        
        assert selected.id == 1


class TestStrategyInterface:


    
    def test_all_strategies_inherit_from_base(self):

        assert issubclass(RandomStrategy, ConflictResolutionStrategy)
        assert issubclass(FirstStrategy, ConflictResolutionStrategy)
        assert issubclass(RecencyStrategy, ConflictResolutionStrategy)
        assert issubclass(SpecificityStrategy, ConflictResolutionStrategy)
    
    def test_all_strategies_have_select_method(self):

        strategies = [RandomStrategy(), FirstStrategy(), RecencyStrategy(), SpecificityStrategy()]
        
        for strategy in strategies:
            assert hasattr(strategy, 'select')
            assert callable(strategy.select)