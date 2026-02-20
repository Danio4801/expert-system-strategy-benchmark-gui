













import pytest
import pandas as pd
from pathlib import Path

from preprocessing.data_loader import DataLoader, CSVConfig
from preprocessing.discretizer import Discretizer
from preprocessing.rule_generator import RuleGenerator
from core.models import Fact, KnowledgeBase
from core.strategies import FirstStrategy, SpecificityStrategy, RandomStrategy
from core.inference import ForwardChaining, GreedyForwardChaining


class TestIrisPipeline:


    
    @pytest.fixture
    def iris_df(self, tmp_path):


        data = """Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
1,5.1,3.5,1.4,0.2,Iris-setosa
2,4.9,3.0,1.4,0.2,Iris-setosa
3,4.7,3.2,1.3,0.2,Iris-setosa
4,7.0,3.2,4.7,1.4,Iris-versicolor
5,6.4,3.2,4.5,1.5,Iris-versicolor
6,6.9,3.1,4.9,1.5,Iris-versicolor
7,6.3,3.3,6.0,2.5,Iris-virginica
8,5.8,2.7,5.1,1.9,Iris-virginica
9,7.1,3.0,5.9,2.1,Iris-virginica
10,6.5,3.0,5.8,2.2,Iris-virginica"""
        
        path = tmp_path / "iris.csv"
        path.write_text(data)
        return path
    
    def test_full_pipeline_loads_csv(self, iris_df):

        loader = DataLoader()
        df = loader.load(iris_df, autodetect=True)
        
        assert len(df) == 10
        assert "Species" in df.columns
    
    def test_full_pipeline_discretizes(self, iris_df):

        loader = DataLoader()
        df = loader.load(iris_df)
        

        df = df.drop(columns=["Id"])
        

        discretizer = Discretizer()
        df_disc = discretizer.discretize(
            df, 
            method="equal_width", 
            bins=3,
            columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        )
        

        assert df_disc["SepalLengthCm"].dtype == object
        assert df_disc["Species"].iloc[0] == "Iris-setosa"
    
    def test_full_pipeline_generates_rules(self, iris_df):

        loader = DataLoader()
        df = loader.load(iris_df)
        df = df.drop(columns=["Id"])
        
        discretizer = Discretizer()
        df_disc = discretizer.discretize(df, method="equal_width", bins=3)
        
        generator = RuleGenerator()
        rules = generator.generate(df_disc, decision_column="Species")
        

        assert len(rules) > 0
        

        assert all(len(r.premises) == 4 for r in rules)
        

        conclusions = {r.conclusion.value for r in rules}
        assert "Iris-setosa" in conclusions
    
    def test_full_pipeline_inference(self, iris_df):


        loader = DataLoader()
        df = loader.load(iris_df)
        df = df.drop(columns=["Id"])
        
        discretizer = Discretizer()
        df_disc = discretizer.discretize(df, method="equal_width", bins=3)
        
        generator = RuleGenerator()
        rules = generator.generate(df_disc, decision_column="Species")
        

        first_row = df_disc.iloc[0]
        initial_facts = {
            Fact("SepalLengthCm", str(first_row["SepalLengthCm"])),
            Fact("SepalWidthCm", str(first_row["SepalWidthCm"])),
            Fact("PetalLengthCm", str(first_row["PetalLengthCm"])),
            Fact("PetalWidthCm", str(first_row["PetalWidthCm"]))
        }
        

        kb = KnowledgeBase(rules=rules, facts=initial_facts)
        

        engine = ForwardChaining(FirstStrategy())
        result = engine.run(kb)
        

        assert result.success == True
        assert len(result.new_facts) > 0
        

        species_facts = [f for f in result.facts if f.attribute == "Species"]
        assert len(species_facts) > 0
    
    def test_full_pipeline_with_goal(self, iris_df):

        loader = DataLoader()
        df = loader.load(iris_df)
        df = df.drop(columns=["Id"])
        
        discretizer = Discretizer()
        df_disc = discretizer.discretize(df, method="equal_width", bins=3)
        
        generator = RuleGenerator()
        rules = generator.generate(df_disc, decision_column="Species")
        

        first_row = df_disc.iloc[0]
        initial_facts = {
            Fact("SepalLengthCm", str(first_row["SepalLengthCm"])),
            Fact("SepalWidthCm", str(first_row["SepalWidthCm"])),
            Fact("PetalLengthCm", str(first_row["PetalLengthCm"])),
            Fact("PetalWidthCm", str(first_row["PetalWidthCm"]))
        }
        
        kb = KnowledgeBase(rules=rules, facts=initial_facts)
        

        goal = Fact("Species", "Iris-setosa")
        
        engine = ForwardChaining(FirstStrategy())
        result = engine.run(kb, goal=goal)
        
        assert result.success == True
    
    def test_compare_strategies(self, iris_df):

        loader = DataLoader()
        df = loader.load(iris_df)
        df = df.drop(columns=["Id"])
        
        discretizer = Discretizer()
        df_disc = discretizer.discretize(df, method="equal_width", bins=3)
        
        generator = RuleGenerator()
        rules = generator.generate(df_disc, decision_column="Species")
        
        first_row = df_disc.iloc[0]
        initial_facts = {
            Fact("SepalLengthCm", str(first_row["SepalLengthCm"])),
            Fact("SepalWidthCm", str(first_row["SepalWidthCm"])),
            Fact("PetalLengthCm", str(first_row["PetalLengthCm"])),
            Fact("PetalWidthCm", str(first_row["PetalWidthCm"]))
        }
        
        strategies = [
            ("First", FirstStrategy()),
            ("Specificity", SpecificityStrategy()),
            ("Random", RandomStrategy())
        ]
        
        results = {}
        for name, strategy in strategies:
            kb = KnowledgeBase(rules=rules.copy(), facts=initial_facts.copy())
            engine = ForwardChaining(strategy)
            result = engine.run(kb)
            results[name] = {
                "iterations": result.iterations,
                "new_facts": len(result.new_facts),
                "time": result.execution_time_ms
            }
        

        assert all(r["new_facts"] > 0 for r in results.values())
        

        print("\n=== Porównanie strategii ===")
        for name, stats in results.items():
            print(f"{name}: iterations={stats['iterations']}, "
                  f"new_facts={stats['new_facts']}, time={stats['time']:.3f}ms")