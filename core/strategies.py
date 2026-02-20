












from abc import ABC, abstractmethod
from typing import List, Set, Dict, Optional
import random

from core.models import Fact, Rule


class ConflictResolutionStrategy(ABC):






    @abstractmethod
    def select(self, conflict_set: List[Rule], facts) -> Rule:













        pass


class RandomStrategy(ConflictResolutionStrategy):











    def __init__(self, seed: Optional[int] = None):








        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

    def select(self, conflict_set: List[Rule], facts: Set[Fact]) -> Rule:










        return self.rng.choice(conflict_set)


class FirstStrategy(ConflictResolutionStrategy):










    def select(self, conflict_set: List[Rule], facts: Set[Fact]) -> Rule:










        return conflict_set[0]


class SpecificityStrategy(ConflictResolutionStrategy):











    def select(self, conflict_set: List[Rule], facts: Set[Fact]) -> Rule:










        return max(conflict_set, key=len)


class RecencyStrategy(ConflictResolutionStrategy):



















    def select(self, conflict_set: List[Rule], facts: Dict[Fact, int]) -> Rule:










        def get_max_recency(rule: Rule) -> int:

            return max(facts[premise] for premise in rule.premises)

        return max(conflict_set, key=get_max_recency)
