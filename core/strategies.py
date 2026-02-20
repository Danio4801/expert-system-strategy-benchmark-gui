"""
Moduł zawierający strategie rozwiązywania konfliktów (conflict resolution).

Gdy wiele reguł może być aktywowanych, strategia wybiera która zostanie wykonana.

Klasy:
    - ConflictResolutionStrategy: abstrakcyjna klasa bazowa
    - RandomStrategy: losowy wybór
    - FirstStrategy: FIFO - pierwsza reguła
    - SpecificityStrategy: reguła z największą liczbą przesłanek
    - RecencyStrategy: reguła używająca najnowszych faktów
"""

from abc import ABC, abstractmethod
from typing import List, Set, Dict, Optional
import random

from core.models import Fact, Rule


class ConflictResolutionStrategy(ABC):
    """
    Abstrakcyjna klasa bazowa dla strategii wyboru reguły z conflict set.

    Wszystkie strategie muszą implementować metodę select().
    """

    @abstractmethod
    def select(self, conflict_set: List[Rule], facts) -> Rule:
        """
        Wybiera jedną regułę z conflict set.

        Args:
            conflict_set: Lista reguł gotowych do aktywacji
            facts: Zbiór faktów (Set[Fact] lub Dict[Fact, int] dla RecencyStrategy)

        Returns:
            Wybrana reguła do wykonania

        Raises:
            ValueError: Gdy conflict_set jest puste
        """
        pass


class RandomStrategy(ConflictResolutionStrategy):
    """
    Strategia wyboru losowego.

    Wybiera losową regułę z conflict set.
    Może przyjąć seed dla deterministycznych wyników (powtarzalność eksperymentów).

    Example:
        >>> strategy = RandomStrategy(seed=42)  # powtarzalne wyniki
        >>> rule = strategy.select(conflict_set, facts)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Tworzy strategię losową.

        Args:
            seed: Opcjonalny seed dla generatora liczb losowych.
                  Jeśli None, używa globalnego random (niepowtarzalne).
                  Jeśli podany, wyniki są deterministyczne (powtarzalne).
        """
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

    def select(self, conflict_set: List[Rule], facts: Set[Fact]) -> Rule:
        """
        Wybiera losową regułę z conflict set.

        Args:
            conflict_set: Lista reguł do wyboru
            facts: Zbiór aktualnych faktów (nie używane w tej strategii)

        Returns:
            Losowo wybrana reguła
        """
        return self.rng.choice(conflict_set)


class FirstStrategy(ConflictResolutionStrategy):
    """
    Strategia FIFO (First In First Out).

    Zawsze wybiera pierwszą regułę z listy (index 0).

    Example:
        >>> strategy = FirstStrategy()
        >>> rule = strategy.select(conflict_set, facts)
    """

    def select(self, conflict_set: List[Rule], facts: Set[Fact]) -> Rule:
        """
        Wybiera pierwszą regułę z conflict set.

        Args:
            conflict_set: Lista reguł do wyboru
            facts: Zbiór aktualnych faktów (nie używane w tej strategii)

        Returns:
            Pierwsza reguła z listy
        """
        return conflict_set[0]


class SpecificityStrategy(ConflictResolutionStrategy):
    """
    Strategia specyficzności.

    Wybiera regułę z największą liczbą przesłanek.
    Przy remisie wybiera pierwszą z najdłuższych.

    Example:
        >>> strategy = SpecificityStrategy()
        >>> rule = strategy.select(conflict_set, facts)
    """

    def select(self, conflict_set: List[Rule], facts: Set[Fact]) -> Rule:
        """
        Wybiera regułę z największą liczbą przesłanek.

        Args:
            conflict_set: Lista reguł do wyboru
            facts: Zbiór aktualnych faktów (nie używane w tej strategii)

        Returns:
            Reguła z największą liczbą przesłanek
        """
        return max(conflict_set, key=len)


class RecencyStrategy(ConflictResolutionStrategy):
    """
    Strategia recencji (nowości) - używa Logical Clock.

    Wybiera regułę która używa najnowszych faktów.
    Dla każdej reguły oblicza maksymalną recency spośród jej przesłanek,
    następnie wybiera regułę o największej recency.

    UWAGA: Ta strategia przyjmuje Dict[Fact, int] zamiast Set[Fact],
    gdzie wartość int to Logical Clock ID (numer iteracji wnioskowania):
    - Initial facts: clock_id = 0
    - New facts: clock_id = iterations (1, 2, 3, ...)
    - Wyższy clock_id = nowszy fakt

    Example:
        >>> strategy = RecencyStrategy()
        >>> facts_with_recency = {Fact("a", "1"): 0, Fact("b", "2"): 5}
        >>> rule = strategy.select(conflict_set, facts_with_recency)
    """

    def select(self, conflict_set: List[Rule], facts: Dict[Fact, int]) -> Rule:
        """
        Wybiera regułę używającą najnowszych faktów.

        Args:
            conflict_set: Lista reguł do wyboru
            facts: Słownik mapujący fakty na ich recency (wyższy = nowszy)

        Returns:
            Reguła o najwyższej maksymalnej recency wśród przesłanek
        """
        def get_max_recency(rule: Rule) -> int:
            """Oblicza maksymalną recency spośród przesłanek reguły."""
            return max(facts[premise] for premise in rule.premises)

        return max(conflict_set, key=get_max_recency)
