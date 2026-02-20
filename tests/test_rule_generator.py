
















import pytest
import pandas as pd

from preprocessing.rule_generator import RuleGenerator
from core.models import Fact, Rule


class TestRuleGeneratorBasic:

    
    def test_generates_rules_from_dataframe(self):

        df = pd.DataFrame({
            "color": ["red", "blue"],
            "size": ["big", "small"],
            "class": ["A", "B"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        
        assert len(rules) == 2
        assert all(isinstance(r, Rule) for r in rules)
    
    def test_rule_has_correct_premises(self):

        df = pd.DataFrame({
            "color": ["red"],
            "size": ["big"],
            "class": ["A"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        
        rule = rules[0]
        assert len(rule.premises) == 2
        assert Fact("color", "red") in rule.premises
        assert Fact("size", "big") in rule.premises
    
    def test_rule_has_correct_conclusion(self):

        df = pd.DataFrame({
            "color": ["red"],
            "size": ["big"],
            "class": ["A"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        
        rule = rules[0]
        assert rule.conclusion == Fact("class", "A")
    
    def test_rules_have_unique_ids(self):

        df = pd.DataFrame({
            "a": ["1", "2", "3"],
            "b": ["x", "y", "z"],
            "class": ["A", "B", "C"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        
        ids = [r.id for r in rules]
        assert len(ids) == len(set(ids))


class TestRuleGeneratorDuplicates:


    
    def test_removes_duplicate_rows(self):

        df = pd.DataFrame({
            "color": ["red", "red", "blue"],
            "size": ["big", "big", "small"],
            "class": ["A", "A", "B"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        

        assert len(rules) == 2
    
    def test_keeps_different_conclusions_for_same_premises(self):


        df = pd.DataFrame({
            "color": ["red", "red"],
            "size": ["big", "big"],
            "class": ["A", "B"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        

        assert len(rules) == 2


class TestRuleGeneratorEdgeCases:


    
    def test_single_premise(self):

        df = pd.DataFrame({
            "color": ["red", "blue"],
            "class": ["A", "B"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        
        assert len(rules) == 2
        assert len(rules[0].premises) == 1
    
    def test_many_premises(self):

        df = pd.DataFrame({
            "a": ["1"],
            "b": ["2"],
            "c": ["3"],
            "d": ["4"],
            "e": ["5"],
            "class": ["X"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        
        assert len(rules[0].premises) == 5
    
    def test_empty_dataframe_raises_error(self):

        df = pd.DataFrame(columns=["a", "b", "class"])
        
        generator = RuleGenerator()
        
        with pytest.raises(ValueError):
            generator.generate(df, decision_column="class")
    
    def test_missing_decision_column_raises_error(self):

        df = pd.DataFrame({
            "a": ["1"],
            "b": ["2"]
        })
        
        generator = RuleGenerator()
        
        with pytest.raises(ValueError):
            generator.generate(df, decision_column="class")
    
    def test_only_decision_column_raises_error(self):

        df = pd.DataFrame({
            "class": ["A", "B"]
        })
        
        generator = RuleGenerator()
        
        with pytest.raises(ValueError):
            generator.generate(df, decision_column="class")


class TestRuleGeneratorDataTypes:


    
    def test_converts_numeric_to_string(self):

        df = pd.DataFrame({
            "age": [25, 30],
            "income": [50000, 60000],
            "class": ["A", "B"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        

        for rule in rules:
            for premise in rule.premises:
                assert isinstance(premise.value, str)
    
    def test_handles_float_values(self):

        df = pd.DataFrame({
            "value": [1.5, 2.7],
            "class": ["A", "B"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        
        assert len(rules) == 2


class TestRuleGeneratorStatistics:

    
    def test_get_statistics(self):

        df = pd.DataFrame({
            "a": ["1", "2", "3"],
            "b": ["x", "y", "z"],
            "class": ["A", "B", "A"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        stats = generator.get_statistics()
        
        assert "total_rules" in stats
        assert "avg_premises" in stats
        assert stats["total_rules"] == 3
        assert stats["avg_premises"] == 2.0



class TestRuleGeneratorExcludeColumns:


    
    def test_exclude_single_column(self):

        df = pd.DataFrame({
            "Id": [1, 2, 3],
            "color": ["red", "blue", "green"],
            "class": ["A", "B", "C"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(
            df, 
            decision_column="class",
            exclude_columns=["Id"]
        )
        

        for rule in rules:
            attributes = [p.attribute for p in rule.premises]
            assert "Id" not in attributes
        

        assert all(len(r.premises) == 1 for r in rules)
    
    def test_exclude_multiple_columns(self):

        df = pd.DataFrame({
            "Id": [1, 2],
            "row_number": [100, 200],
            "color": ["red", "blue"],
            "class": ["A", "B"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(
            df,
            decision_column="class",
            exclude_columns=["Id", "row_number"]
        )
        
        for rule in rules:
            attributes = [p.attribute for p in rule.premises]
            assert "Id" not in attributes
            assert "row_number" not in attributes
    
    def test_exclude_nonexistent_column_ignored(self):

        df = pd.DataFrame({
            "color": ["red", "blue"],
            "class": ["A", "B"]
        })
        
        generator = RuleGenerator()

        rules = generator.generate(
            df,
            decision_column="class",
            exclude_columns=["nonexistent"]
        )
        
        assert len(rules) == 2


class TestRuleGeneratorAutoDetectId:


    
    def test_detect_column_named_id(self):

        df = pd.DataFrame({
            "Id": [1, 2, 3],
            "color": ["red", "blue", "green"],
            "class": ["A", "B", "C"]
        })
        
        generator = RuleGenerator()
        id_columns = generator.detect_id_columns(df)
        
        assert "Id" in id_columns
    
    def test_detect_column_named_index(self):

        df = pd.DataFrame({
            "row_index": [1, 2, 3],
            "color": ["red", "blue", "green"],
            "class": ["A", "B", "C"]
        })
        
        generator = RuleGenerator()
        id_columns = generator.detect_id_columns(df)
        
        assert "row_index" in id_columns
    
    def test_detect_sequential_unique_column(self):

        df = pd.DataFrame({
            "row_nr": [1, 2, 3, 4, 5],
            "color": ["red", "blue", "green", "red", "blue"],
            "class": ["A", "B", "C", "A", "B"]
        })
        
        generator = RuleGenerator()
        id_columns = generator.detect_id_columns(df)
        
        assert "row_nr" in id_columns
    
    def test_does_not_detect_normal_numeric_column(self):

        df = pd.DataFrame({
            "age": [25, 30, 25, 40, 30],
            "color": ["red", "blue", "green", "red", "blue"],
            "class": ["A", "B", "C", "A", "B"]
        })
        
        generator = RuleGenerator()
        id_columns = generator.detect_id_columns(df)
        
        assert "age" not in id_columns
    
    def test_auto_exclude_id_columns(self):

        df = pd.DataFrame({
            "Id": [1, 2, 3],
            "color": ["red", "blue", "green"],
            "class": ["A", "B", "C"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(
            df,
            decision_column="class",
            auto_exclude_id=True
        )
        

        for rule in rules:
            attributes = [p.attribute for p in rule.premises]
            assert "Id" not in attributes
    
    def test_auto_exclude_disabled_by_default(self):

        df = pd.DataFrame({
            "Id": [1, 2, 3],
            "color": ["red", "blue", "green"],
            "class": ["A", "B", "C"]
        })
        
        generator = RuleGenerator()
        rules = generator.generate(df, decision_column="class")
        

        all_attributes = []
        for rule in rules:
            all_attributes.extend([p.attribute for p in rule.premises])
        
        assert "Id" in all_attributes