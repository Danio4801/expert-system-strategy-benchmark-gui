






from typing import List, Optional
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import logging
import random

from core.models import Fact, Rule


default_logger = logging.getLogger(__name__)


class ForestRuleGenerator:















    def __init__(self, n_estimators: int = 5, max_depth: int = 5, min_samples_leaf: int = 5, random_state: int = 42, logger: Optional[logging.Logger] = None, min_depth: int = 2):











        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.encoder = None
        self.estimators_ = []
        self.feature_names = None
        self.class_names = None
        self.decision_column = None
        self.rule_id_counter = 0
        self.logger = logger if logger else default_logger
        self.rng = random.Random(random_state)

    def generate(self, df: pd.DataFrame, decision_column: str) -> List[Rule]:










        self.decision_column = decision_column
        self.rule_id_counter = 0


        X = df.drop(columns=[decision_column])
        y = df[decision_column]

        self.feature_names = list(X.columns)
        self.class_names = y.unique().tolist()


        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded = self.encoder.fit_transform(X)


        self.logger.debug(f"[FOREST] Starting Variable-Depth Forest training with {self.n_estimators} trees (depth range: [{self.min_depth}, {self.max_depth}])")
        self.logger.debug(f"[FOREST] Training on {len(X)} samples with {len(self.feature_names)} features")


        self.estimators_ = []
        all_rules = []

        for i in range(1, self.n_estimators + 1):

            current_depth = self.rng.randint(self.min_depth, self.max_depth)


            tree = DecisionTreeClassifier(
                max_depth=current_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",
                random_state=self.random_state + i
            )
            tree.fit(X_encoded, y)
            self.estimators_.append(tree)


            self.logger.debug(f"[FOREST] Tree {i}/{self.n_estimators} trained with max_depth={current_depth}")


            rules_from_tree = self._extract_rules_from_single_tree(tree)
            all_rules.extend(rules_from_tree)


            self.logger.debug(f"[FOREST] Tree {i}/{self.n_estimators} (depth={current_depth}) yielded {len(rules_from_tree)} rules (total so far: {len(all_rules)})")


        self.logger.info(f"[FOREST] Rule generation completed: {len(all_rules)} total rules from {self.n_estimators} trees")
        if len(all_rules) > 0:
            avg_rules_per_tree = len(all_rules) / self.n_estimators
            depth_distribution = [tree.get_depth() for tree in self.estimators_]
            avg_depth = sum(depth_distribution) / len(depth_distribution)
            self.logger.debug(f"[FOREST] Average rules per tree: {avg_rules_per_tree:.2f}, Average actual depth: {avg_depth:.2f}")

        return all_rules

    def _extract_rules_from_single_tree(self, tree_model) -> List[Rule]:









        tree = tree_model.tree_
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value

        rules = []


        def traverse(node_id: int, current_premises: List[Fact]):



            if tree.children_left[node_id] == tree.children_right[node_id]:


                class_counts = value[node_id][0]
                predicted_class_idx = np.argmax(class_counts)


                predicted_class = tree_model.classes_[predicted_class_idx]


                conclusion = Fact(self.decision_column, str(predicted_class))


                if current_premises:
                    rule = Rule(
                        id=self.rule_id_counter,
                        premises=current_premises.copy(),
                        conclusion=conclusion
                    )
                    rules.append(rule)
                    self.rule_id_counter += 1

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
