











from typing import List
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

from core.models import Fact, Rule


class TreeRuleGenerator:













    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 5, random_state: int = 42):










        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.encoder = None
        self.tree_model = None
        self.feature_names = None
        self.class_names = None
        self.decision_column = None

    def generate(self, df: pd.DataFrame, decision_column: str) -> List[Rule]:










        self.decision_column = decision_column


        X = df.drop(columns=[decision_column])
        y = df[decision_column]

        self.feature_names = list(X.columns)
        self.class_names = y.unique().tolist()


        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded = self.encoder.fit_transform(X)


        self.tree_model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.tree_model.fit(X_encoded, y)


        rules = self._extract_rules_from_tree()

        return rules

    def _extract_rules_from_tree(self) -> List[Rule]:






        tree = self.tree_model.tree_
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value

        rules = []
        rule_id = 0


        def traverse(node_id: int, current_premises: List[Fact]):



            if tree.children_left[node_id] == tree.children_right[node_id]:


                class_counts = value[node_id][0]
                predicted_class_idx = np.argmax(class_counts)
                predicted_class = self.tree_model.classes_[predicted_class_idx]


                conclusion = Fact(self.decision_column, str(predicted_class))


                if current_premises:
                    nonlocal rule_id
                    rule = Rule(
                        id=rule_id,
                        premises=current_premises.copy(),
                        conclusion=conclusion
                    )
                    rules.append(rule)
                    rule_id += 1

                return


            feature_idx = feature[node_id]
            feature_name = self.feature_names[feature_idx]
            threshold_value = threshold[node_id]






            left_category_idx = int(np.floor(threshold_value))
            if left_category_idx >= 0:
                try:

                    left_category = self.encoder.categories_[feature_idx][left_category_idx]
                    left_premise = Fact(feature_name, str(left_category))
                    traverse(tree.children_left[node_id], current_premises + [left_premise])
                except (IndexError, KeyError):

                    traverse(tree.children_left[node_id], current_premises)


            right_category_idx = int(np.ceil(threshold_value))
            if right_category_idx < len(self.encoder.categories_[feature_idx]):
                try:
                    right_category = self.encoder.categories_[feature_idx][right_category_idx]
                    right_premise = Fact(feature_name, str(right_category))
                    traverse(tree.children_right[node_id], current_premises + [right_premise])
                except (IndexError, KeyError):

                    traverse(tree.children_right[node_id], current_premises)


        traverse(0, [])

        return rules

    def get_tree_stats(self) -> dict:






        if self.tree_model is None:
            return {}

        return {
            'n_leaves': self.tree_model.get_n_leaves(),
            'max_depth': self.tree_model.get_depth(),
            'n_nodes': self.tree_model.tree_.node_count,
            'n_features': len(self.feature_names),
            'n_classes': len(self.class_names)
        }


def compare_rule_generators(df: pd.DataFrame, decision_column: str):









    from preprocessing.rule_generator import RuleGenerator

    print("=" * 80)
    print("COMPARISON: Naive vs Tree-based Rule Generation")
    print("=" * 80)
    print()


    print("1. NAIVE RULE GENERATOR (1 row = 1 rule)")
    print("-" * 80)
    naive_gen = RuleGenerator()
    naive_rules = naive_gen.generate(df, decision_column)
    print(f"   Total rules: {len(naive_rules)}")
    if naive_rules:
        avg_premises = sum(len(r.premises) for r in naive_rules) / len(naive_rules)
        print(f"   Avg premises per rule: {avg_premises:.1f}")
        print(f"\n   Example rule:")
        print(f"   {naive_rules[0]}")
    print()


    print("2. TREE-BASED RULE GENERATOR (paths from tree)")
    print("-" * 80)
    tree_gen = TreeRuleGenerator(max_depth=5, min_samples_leaf=10)
    tree_rules = tree_gen.generate(df, decision_column)
    print(f"   Total rules: {len(tree_rules)}")
    if tree_rules:
        avg_premises = sum(len(r.premises) for r in tree_rules) / len(tree_rules)
        print(f"   Avg premises per rule: {avg_premises:.1f}")
        print(f"\n   Example rules:")
        for i, rule in enumerate(tree_rules[:3], 1):
            print(f"   {i}. {rule}")
    print()

    stats = tree_gen.get_tree_stats()
    print(f"   Tree stats: {stats}")
    print()

    print("=" * 80)
    print("KEY DIFFERENCE:")
    print("  Naive: Many rules, very specific (hard to activate)")
    print("  Tree:  Fewer rules, more general (easy to activate, more conflicts)")
    print("=" * 80)
