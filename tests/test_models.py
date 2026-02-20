
















import pytest
from core.models import Fact, Rule, KnowledgeBase

class TestFact:

    
    def test_create_fact(self):

        fact = Fact("kolor", "czerwony")
        assert fact.attribute == "kolor"
        assert fact.value == "czerwony"
    
    def test_facts_with_same_values_are_equal(self):


        fact1 = Fact("kolor", "czerwony")
        fact2 = Fact("kolor", "czerwony")
        assert fact1 == fact2
    
    def test_facts_with_different_values_are_not_equal(self):

        fact1 = Fact("kolor", "czerwony")
        fact2 = Fact("kolor", "niebieski")
        assert fact1 != fact2
    
    def test_facts_with_different_attributes_are_not_equal(self):

        fact1 = Fact("kolor", "czerwony")
        fact2 = Fact("rozmiar", "czerwony")
        assert fact1 != fact2
    
    def test_fact_can_be_in_set(self):


        fact1 = Fact("kolor", "czerwony")
        fact2 = Fact("kolor", "czerwony")
        fact3 = Fact("rozmiar", "duży")
        
        fact_set = {fact1, fact2, fact3}
        assert len(fact_set) == 2
    
    def test_fact_repr(self):

        fact = Fact("kolor", "czerwony")
        repr_str = repr(fact)
        assert "kolor" in repr_str
        assert "czerwony" in repr_str
    
    def test_empty_attribute_raises_error(self):

        with pytest.raises(ValueError):
            Fact("", "czerwony")
    
    def test_empty_value_raises_error(self):

        with pytest.raises(ValueError):
            Fact("kolor", "")




class TestRule:


    def test_create_rule(self):

        premises = [Fact("kolor", "czerwony"), Fact("rozmiar", "duży")]
        conclusion = Fact("klasa", "A")
        rule = Rule(id=1, premises=premises, conclusion=conclusion)
        
        assert rule.id == 1
        assert len(rule.premises) == 2
        assert rule.conclusion == conclusion
    
    def test_rule_is_satisfied_when_all_premises_in_facts(self):



        premises = [Fact("kolor", "czerwony"), Fact("rozmiar", "duży")]
        conclusion = Fact("klasa", "A")
        rule = Rule(id=1, premises=premises, conclusion=conclusion)
        
        facts = {Fact("kolor", "czerwony"), Fact("rozmiar", "duży"), Fact("waga", "lekka")}
        assert rule.is_satisfied_by(facts) == True
    
    def test_rule_is_not_satisfied_when_premise_missing(self):


        premises = [Fact("kolor", "czerwony"), Fact("rozmiar", "duży")]
        conclusion = Fact("klasa", "A")
        rule = Rule(id=1, premises=premises, conclusion=conclusion)
        
        facts = {Fact("kolor", "czerwony")}
        assert rule.is_satisfied_by(facts) == False
    
    def test_rule_len_returns_number_of_premises(self):


        premises = [Fact("a", "1"), Fact("b", "2"), Fact("c", "3")]
        conclusion = Fact("d", "4")
        rule = Rule(id=1, premises=premises, conclusion=conclusion)
        
        assert len(rule) == 3
    
    def test_rule_repr_contains_premises_and_conclusion(self):

        premises = [Fact("kolor", "czerwony")]
        conclusion = Fact("klasa", "A")
        rule = Rule(id=1, premises=premises, conclusion=conclusion)
        
        repr_str = repr(rule)
        assert "kolor" in repr_str
        assert "klasa" in repr_str
    
    def test_empty_premises_raises_error(self):

        with pytest.raises(ValueError):
            Rule(id=1, premises=[], conclusion=Fact("klasa", "A"))
    
    def test_negative_id_raises_error(self):

        with pytest.raises(ValueError):
            Rule(id=-1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))




class TestKnowledgeBase:



    def test_create_empty_knowledge_base(self):

        kb = KnowledgeBase()
        assert len(kb.rules) == 0
        assert len(kb.facts) == 0
    
    def test_create_knowledge_base_with_rules_and_facts(self):

        rules = [Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))]
        facts = {Fact("a", "1")}
        kb = KnowledgeBase(rules=rules, facts=facts)
        
        assert len(kb.rules) == 1
        assert len(kb.facts) == 1
    
    def test_add_fact(self):

        kb = KnowledgeBase()
        kb.add_fact(Fact("kolor", "czerwony"))
        
        assert len(kb.facts) == 1
        assert kb.has_fact(Fact("kolor", "czerwony"))
    
    def test_add_facts(self):

        kb = KnowledgeBase()
        kb.add_facts([Fact("a", "1"), Fact("b", "2")])
        
        assert len(kb.facts) == 2
    
    def test_has_fact_returns_true_when_fact_exists(self):

        kb = KnowledgeBase(facts={Fact("kolor", "czerwony")})
        assert kb.has_fact(Fact("kolor", "czerwony")) == True
    
    def test_has_fact_returns_false_when_fact_missing(self):

        kb = KnowledgeBase()
        assert kb.has_fact(Fact("kolor", "czerwony")) == False
    
    def test_get_applicable_rules_returns_satisfied_rules(self):


        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        rule2 = Rule(id=2, premises=[Fact("c", "3")], conclusion=Fact("d", "4"))
        
        kb = KnowledgeBase(rules=[rule1, rule2], facts={Fact("a", "1")})
        
        applicable = kb.get_applicable_rules()
        assert len(applicable) == 1
        assert applicable[0].id == 1
    
    def test_get_applicable_rules_excludes_rules_with_conclusion_in_facts(self):


        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        

        kb = KnowledgeBase(rules=[rule1], facts={Fact("a", "1"), Fact("b", "2")})
        
        applicable = kb.get_applicable_rules()
        assert len(applicable) == 0
    
    def test_get_applicable_rules_returns_empty_when_no_rules_match(self):

        rule1 = Rule(id=1, premises=[Fact("a", "1")], conclusion=Fact("b", "2"))
        
        kb = KnowledgeBase(rules=[rule1], facts={Fact("x", "y")})
        
        applicable = kb.get_applicable_rules()
        assert len(applicable) == 0
