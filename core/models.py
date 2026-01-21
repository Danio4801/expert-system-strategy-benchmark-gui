"""
To jest moduł zawierający podstawowe modele danych systemu ekspertowego zwanego dalej SE.

Klasy:
    - Fact: reprezentuje pojedynczy fakt (para atrybut-wartość), jest to atomowa jednostka informacji
    - Rule: reprezentuje regułę IF-THEN
    - KnowledgeBase: przechowuje reguły i fakty
"""

from typing import List, Set, Iterable


class Fact:
    """
    Klasa reprezentuje pojedynczy fakt w systemie ekspertowym.

    Fakt to para (atrybut, wartość), np. ("goraczka", "tak") albo ("auto", "szybkie")
    Klasa jest niemodyfikowalna (w sensie logicznym) i haszowalna,
    co pozwala na używanie jej w zbiorach (set).

    Attributes:
        attribute (str): Nazwa atrybutu
        value (str): Wartość atrybutu

    Example:
        fact = Fact("temperatura", "wysoka")

        fact.attribute
        'temperatura'

        fact.value
        'wysoka'
    """

    def __init__(self, attribute: str, value: str):
        """
        Klasa tworzy nowy fakt.

        Args:
            attribute: Nazwa atrybutu (nie może być pusta)
            value: Wartość atrybutu (nie może być pusta)

        Raises:
            ValueError: Gdy attribute lub value jest puste
        """
        if not attribute:
            raise ValueError("Attribute cannot be empty")
        if not value:
            raise ValueError("Value cannot be empty")

        self.attribute = attribute
        self.value = value

    def __eq__(self, other):
        """
        Funkcja sprawdza czy dwa fakty są równe.
        Fakty są równe jeśli mają ten sam atrybut i wartość.
        """
        if not isinstance(other, Fact):
            return False
        return self.attribute == other.attribute and self.value == other.value

    def __hash__(self):
        """
        Funkcja zwraca hash faktu (który jest nam potrzebny do przechowywania w zbiorze).
        """
        return hash((self.attribute, self.value))

    def __repr__(self):
        """
        Funkcja zwraca reprezentację tekstową faktu.
        """
        return f"Fact({self.attribute}={self.value})"


class Rule:
    """
    Klasa reprezentuje regułę IF-THEN w systemie ekspertowym.

    Reguła składa się z:
    - przesłanek (premises): lista faktów które muszą być spełnione
    - konkluzji (conclusion): fakt który zostanie dodany gdy przesłanki spełnione

    Example:
        premises = [Fact("gorączka", "tak"), Fact("drapiący kaszel", "tak")]
        conclusion = Fact("diagnoza", "grypa")

        rule = Rule(id=1, premises=premises, conclusion=conclusion)
        len(rule)  # liczba przesłanek
        2
    """

    def __init__(self, id: int, premises: List[Fact], conclusion: Fact):
        """
        Tworzy nową regułę.

        Args:
            id: Identyfikator reguły (musi być >= 0)
            premises: Lista przesłanek (nie może być pusta)
            conclusion: Konkluzja reguły

        Raises:
            ValueError: Gdy id < 0 lub premises jest puste lub conclusion jest None
        """
        if id < 0:
            raise ValueError("Rule ID must be non-negative")
        if not premises:
            raise ValueError("Premises cannot be empty")
        if conclusion is None:
            raise ValueError("Conclusion cannot be None")

        self.id = id
        self.premises = premises
        self.conclusion = conclusion

    def is_satisfied_by(self, facts: Set[Fact]) -> bool:
        """
        Sprawdza czy wszystkie przesłanki reguły są zawarte w podanym zbiorze faktów.

        Args:
            facts: Zbiór faktów do sprawdzenia

        Returns:
            True jeśli wszystkie przesłanki są w facts, False w przeciwnym razie
        """
        return all(premise in facts for premise in self.premises)

    def __len__(self):
        """
        Zwraca liczbę przesłanek w regule.
        """
        return len(self.premises)

    def __repr__(self):
        """
        Zwraca czytelną reprezentację tekstową reguły.
        """
        premises_str = " AND ".join(f"{p.attribute}={p.value}" for p in self.premises)
        conclusion_str = f"{self.conclusion.attribute}={self.conclusion.value}"
        return f"Rule({self.id}): IF {premises_str} THEN {conclusion_str}"


class KnowledgeBase:
    """
    Baza wiedzy przechowująca reguły i fakty.

    Attributes:
        rules (List[Rule]): Lista reguł
        facts (Set[Fact]): Zbiór znanych faktów

    Example:
            kb = KnowledgeBase()
            kb.add_fact(Fact("ton", "niski"))
            kb.has_fact(Fact("ton", "niski"))
        True
    """

    def __init__(self, rules: List[Rule] = None, facts: Set[Fact] = None):
        """
        Tworzy nową bazę wiedzy.

        Args:
            rules: Opcjonalna lista reguł (jeśli None, tworzy pustą listę)
            facts: Opcjonalny zbiór faktów (jeśli None, tworzy pusty zbiór)
        """
        self.rules = rules if rules is not None else []
        self.facts = facts if facts is not None else set()

    def add_fact(self, fact: Fact) -> None:
        """
        Dodaje pojedynczy fakt do bazy.

        Args:
            fact: Fakt do dodania
        """
        self.facts.add(fact)

    def add_facts(self, facts: Iterable[Fact]) -> None:
        """
        Dodaje wiele faktów do bazy.

        Args:
            facts: Iterable zawierające fakty do dodania
        """
        self.facts.update(facts)

    def has_fact(self, fact: Fact) -> bool:
        """
        Sprawdza czy fakt istnieje w bazie.

        Args:
            fact: Fakt do sprawdzenia

        Returns:
            True jeśli fakt istnieje, False w przeciwnym razie
        """
        return fact in self.facts

    def get_applicable_rules(self) -> List[Rule]:
        """
        Zwraca listę reguł które mogą być aktywowane (conflict set).

        Reguła może być aktywowana gdy:
        1. Wszystkie jej przesłanki są spełnione (są w facts)
        2. Jej konkluzja NIE jest jeszcze w facts

        Returns:
            Lista reguł gotowych do aktywacji
        """
        return [
            rule for rule in self.rules
            if rule.is_satisfied_by(self.facts) and rule.conclusion not in self.facts
        ]
