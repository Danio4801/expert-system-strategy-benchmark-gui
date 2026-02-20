








from typing import List, Set, Iterable


class Fact:


















    def __init__(self, attribute: str, value: str):










        if not attribute:
            raise ValueError("Attribute cannot be empty")
        if not value:
            raise ValueError("Value cannot be empty")

        self.attribute = attribute
        self.value = value

    def __eq__(self, other):





        if not isinstance(other, Fact):
            return False
        return self.attribute == other.attribute and self.value == other.value

    def __hash__(self):



        return hash((self.attribute, self.value))

    def __repr__(self):



        return f"Fact({self.attribute}={self.value})"


class Rule:















    def __init__(self, id: int, premises: List[Fact], conclusion: Fact):











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









        return all(premise in facts for premise in self.premises)

    def __len__(self):



        return len(self.premises)

    def __repr__(self):



        premises_str = " AND ".join(f"{p.attribute}={p.value}" for p in self.premises)
        conclusion_str = f"{self.conclusion.attribute}={self.conclusion.value}"
        return f"Rule({self.id}): IF {premises_str} THEN {conclusion_str}"


class KnowledgeBase:














    def __init__(self, rules: List[Rule] = None, facts: Set[Fact] = None):







        self.rules = rules if rules is not None else []
        self.facts = facts if facts is not None else set()

    def add_fact(self, fact: Fact) -> None:






        self.facts.add(fact)

    def add_facts(self, facts: Iterable[Fact]) -> None:






        self.facts.update(facts)

    def has_fact(self, fact: Fact) -> bool:









        return fact in self.facts

    def get_applicable_rules(self) -> List[Rule]:










        return [
            rule for rule in self.rules
            if rule.is_satisfied_by(self.facts) and rule.conclusion not in self.facts
        ]
