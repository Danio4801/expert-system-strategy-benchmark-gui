




















import pytest
import numpy as np
from core.models import Rule, Fact, KnowledgeBase
from core.clustering import RuleClusterer, RuleCluster
from core.inference_clustered import ClusteredForwardChaining
from core.strategies import FirstStrategy






def test_rule_clusterer_determinism():






    rules = [
        Rule(1, [Fact("A", "1"), Fact("B", "1")], Fact("Z", "1")),
        Rule(2, [Fact("A", "1"), Fact("B", "2")], Fact("Z", "1")),
        Rule(3, [Fact("A", "2"), Fact("B", "1")], Fact("Z", "1")),
        Rule(4, [Fact("C", "1"), Fact("D", "1")], Fact("Z", "2")),
        Rule(5, [Fact("C", "1"), Fact("D", "2")], Fact("Z", "2")),
        Rule(6, [Fact("C", "2"), Fact("D", "1")], Fact("Z", "2")),
        Rule(7, [Fact("E", "1"), Fact("F", "1")], Fact("Z", "3")),
        Rule(8, [Fact("E", "1"), Fact("F", "2")], Fact("Z", "3")),
        Rule(9, [Fact("E", "2"), Fact("F", "1")], Fact("Z", "3")),
        Rule(10, [Fact("G", "1")], Fact("Z", "4")),
    ]


    clusterer1 = RuleClusterer(n_clusters=3, random_state=42, method='kmeans')
    clusters1 = clusterer1.fit(rules)


    clusterer2 = RuleClusterer(n_clusters=3, random_state=42, method='kmeans')
    clusters2 = clusterer2.fit(rules)



    for c1, c2 in zip(clusters1, clusters2):
        ids1 = sorted([r.id for r in c1.rules])
        ids2 = sorted([r.id for r in c2.rules])
        assert ids1 == ids2, f"Cluster {c1.cluster_id}: różne przypisania przy tym samym seedzie!"


def test_rule_clusterer_different_seeds_give_different_results():



    rules = [
        Rule(i, [Fact(f"A{i%3}", str(i%5))], Fact("Z", str(i%2)))
        for i in range(20)
    ]


    clusters1 = RuleClusterer(n_clusters=4, random_state=42, method='kmeans').fit(rules)
    clusters2 = RuleClusterer(n_clusters=4, random_state=99, method='kmeans').fit(rules)


    assignments1 = {}
    for cluster in clusters1:
        for rule in cluster.rules:
            assignments1[rule.id] = cluster.cluster_id

    assignments2 = {}
    for cluster in clusters2:
        for rule in cluster.rules:
            assignments2[rule.id] = cluster.cluster_id


    different_count = sum(1 for rid in assignments1 if assignments1[rid] != assignments2[rid])


    assert different_count > 0, "Różne seedy powinny dawać różne klastry!"






def test_rule_clusterer_single_rule_cluster():




    rule = Rule(1, [Fact("A", "1"), Fact("B", "2")], Fact("Z", "X"))


    clusterer = RuleClusterer(n_clusters=1)
    centroid = clusterer._compute_centroid([rule], cluster_id=0)


    centroid_facts = set((p.attribute, p.value) for p in centroid.premises)
    rule_facts = set((p.attribute, p.value) for p in rule.premises)

    assert centroid_facts == rule_facts, "Centroid klastra z 1 regułą musi być identyczny!"
    assert centroid.conclusion == rule.conclusion, "Konkluzja musi być taka sama!"


def test_rule_clusterer_empty_list():



    clusterer = RuleClusterer(n_clusters=5)
    clusters = clusterer.fit([])

    assert clusters == [], "Pusta lista reguł powinna zwrócić puste klastry!"


def test_rule_clusterer_fewer_rules_than_clusters():



    rules = [
        Rule(1, [Fact("A", "1")], Fact("Z", "1")),
        Rule(2, [Fact("B", "1")], Fact("Z", "2")),
    ]


    clusterer = RuleClusterer(n_clusters=10)
    clusters = clusterer.fit(rules)


    assert len(clusters) <= 2, f"Nie można mieć więcej klastrów niż reguł! Got: {len(clusters)}"






def test_clustered_fc_argmax_selects_best_cluster():









    rules1 = [Rule(1, [Fact(f"A{i}", "1") for i in range(10)], Fact("Z", "1"))]
    centroid1 = Rule(1000000, [Fact(f"A{i}", "1") for i in range(10)], Fact("Z", "1"))
    cluster1 = RuleCluster(cluster_id=0, rules=rules1, centroid=centroid1, size=1)


    rules2 = [Rule(2, [Fact(f"A{i}", "1") for i in range(5)], Fact("Z", "2"))]
    centroid2 = Rule(1000001, [Fact(f"A{i}", "1") for i in range(5)], Fact("Z", "2"))
    cluster2 = RuleCluster(cluster_id=1, rules=rules2, centroid=centroid2, size=1)


    initial_facts = {Fact(f"A{i}", "1") for i in range(8)}

    kb = KnowledgeBase(rules=rules1 + rules2, facts=initial_facts)

    engine = ClusteredForwardChaining(
        FirstStrategy(),
        clusters=[cluster1, cluster2],
        centroid_match_threshold=0.0
    )

    result = engine.run(kb)




    assert engine.clusters_skipped >= engine.clusters_checked, \
        f"W argmax pomijamy co najmniej tyle klastrów ile sprawdzamy! Checked: {engine.clusters_checked}, Skipped: {engine.clusters_skipped}"
    assert result.rules_evaluated >= 1, "Reguła w najlepszym klastrze powinna być sprawdzona"


def test_clustered_fc_argmax_stops_when_max_similarity_zero():






    rules = [Rule(1, [Fact("X", "9")], Fact("Z", "1"))]
    centroid = Rule(1000000, [Fact("X", "9")], Fact("Z", "1"))
    cluster = RuleCluster(cluster_id=0, rules=rules, centroid=centroid, size=1)


    initial_facts = {Fact("A", "1"), Fact("B", "2")}

    kb = KnowledgeBase(rules=rules, facts=initial_facts)

    engine = ClusteredForwardChaining(
        FirstStrategy(),
        clusters=[cluster],
        centroid_match_threshold=0.0
    )

    result = engine.run(kb)



    assert engine.clusters_checked == 0, f"Żaden klaster nie powinien być sprawdzony (max_sim=0)! Got: {engine.clusters_checked}"
    assert engine.clusters_skipped == 1, f"Klaster powinien być pominięty! Got: {engine.clusters_skipped}"
    assert result.rules_evaluated == 0, f"Żadna reguła nie powinna być sprawdzona! Got: {result.rules_evaluated}"


def test_clustered_fc_argmax_with_single_cluster():



    rule = Rule(1, [Fact("A", "1"), Fact("B", "1")], Fact("Z", "1"))
    centroid = Rule(1000000, [Fact("A", "1")], Fact("Z", "1"))
    cluster = RuleCluster(cluster_id=0, rules=[rule], centroid=centroid, size=1)


    kb = KnowledgeBase(rules=[rule], facts={Fact("A", "1")})

    engine = ClusteredForwardChaining(
        FirstStrategy(),
        clusters=[cluster],
        centroid_match_threshold=0.0
    )

    result = engine.run(kb)


    assert engine.clusters_checked == 1, "Klaster powinien być sprawdzony (argmax)!"
    assert result.rules_evaluated == 1, "Reguła powinna być sprawdzona"






def test_clustered_fc_clustering_stats():








    clusters = [
        RuleCluster(
            cluster_id=0,
            rules=[Rule(1, [Fact("A", "1")], Fact("Z", "1"))],
            centroid=Rule(1000000, [Fact("A", "1")], Fact("Z", "1")),
            size=1
        ),
        RuleCluster(
            cluster_id=1,
            rules=[Rule(2, [Fact("B", "1"), Fact("X", "9")], Fact("Z", "2"))],
            centroid=Rule(1000001, [Fact("B", "1")], Fact("Z", "2")),
            size=1
        ),
        RuleCluster(
            cluster_id=2,
            rules=[Rule(3, [Fact("C", "1")], Fact("Z", "3"))],
            centroid=Rule(1000002, [Fact("C", "1")], Fact("Z", "3")),
            size=1
        ),
    ]


    facts = {Fact("A", "1"), Fact("B", "1")}

    kb = KnowledgeBase(
        rules=[c.rules[0] for c in clusters],
        facts=facts
    )

    engine = ClusteredForwardChaining(
        FirstStrategy(),
        clusters=clusters,
        centroid_match_threshold=0.0
    )

    result = engine.run(kb)
    stats = engine.get_clustering_stats()




    assert stats['total_clusters'] == 3
    assert stats['clusters_checked'] >= 1, f"Przynajmniej 1 klaster powinien być sprawdzony, got {stats['clusters_checked']}"

    assert stats['clusters_skipped'] >= 2, f"W argmax pomijamy n-1 klastrów na iterację, got {stats['clusters_skipped']}"
    assert stats['algorithm'] == 'argmax (Algorithm 2)'






def test_rule_clusterer_centroid_most_common_facts():




    rules = [
        Rule(1, [Fact("A", "1"), Fact("B", "1"), Fact("X", "1")], Fact("Z", "1")),
        Rule(2, [Fact("A", "1"), Fact("B", "1"), Fact("Y", "2")], Fact("Z", "1")),
        Rule(3, [Fact("A", "1"), Fact("B", "2"), Fact("Z", "3")], Fact("Z", "1")),
    ]


    clusterer = RuleClusterer(n_clusters=1)
    centroid = clusterer._compute_centroid(rules, cluster_id=0)


    centroid_facts = set((p.attribute, p.value) for p in centroid.premises)

    assert ("A", "1") in centroid_facts, "Najbardziej popularny fakt musi być w centroidzie!"


def test_rule_clusterer_centroid_conclusion_most_common():




    rules = [
        Rule(1, [Fact("A", "1")], Fact("Z", "1")),
        Rule(2, [Fact("B", "1")], Fact("Z", "1")),
        Rule(3, [Fact("C", "1")], Fact("Z", "2")),
    ]

    clusterer = RuleClusterer(n_clusters=1)
    centroid = clusterer._compute_centroid(rules, cluster_id=0)


    assert centroid.conclusion == Fact("Z", "1"), "Centroid powinien mieć najczęstszą konkluzję!"






def test_full_pipeline_clustering_and_inference():









    rules = [

        Rule(1, [Fact("A", "1"), Fact("B", "1")], Fact("Z", "1")),
        Rule(2, [Fact("A", "1"), Fact("B", "2")], Fact("Z", "1")),

        Rule(3, [Fact("C", "1"), Fact("D", "1")], Fact("Z", "2")),
        Rule(4, [Fact("C", "1"), Fact("D", "2")], Fact("Z", "2")),

        Rule(5, [Fact("E", "1"), Fact("F", "1")], Fact("Z", "3")),
        Rule(6, [Fact("E", "1"), Fact("F", "2")], Fact("Z", "3")),
    ]


    clusterer = RuleClusterer(n_clusters=3, random_state=42)
    clusters = clusterer.fit(rules)


    facts = {Fact("A", "1"), Fact("B", "1")}

    kb = KnowledgeBase(rules=rules, facts=facts)


    engine = ClusteredForwardChaining(
        FirstStrategy(),
        clusters=clusters,
        centroid_match_threshold=0.0
    )

    result = engine.run(kb)





    assert engine.clusters_checked >= 1, "Przynajmniej jeden klaster powinien być sprawdzony!"
    assert engine.clusters_skipped >= engine.clusters_checked * 2, \
        f"W argmax pomijamy (n-1) klastrów na iterację! Checked: {engine.clusters_checked}, Skipped: {engine.clusters_skipped}"

    assert result.rules_evaluated > 0, "Przynajmniej jedna reguła powinna być sprawdzona!"






def test_centroid_method_general():












    rules = [
        Rule(1, [Fact("a", "1"), Fact("b", "2"), Fact("c", "3")], Fact("dec", "A")),
        Rule(2, [Fact("a", "1"), Fact("b", "2"), Fact("d", "4")], Fact("dec", "A")),
        Rule(3, [Fact("a", "1"), Fact("e", "5")], Fact("dec", "A")),
    ]

    clusterer = RuleClusterer(n_clusters=1, centroid_method='general')
    centroid = clusterer._compute_centroid(rules, cluster_id=0)

    centroid_facts = set((p.attribute, p.value) for p in centroid.premises)


    assert centroid_facts == {("a", "1")}, \
        f"Profile_general powinien zawierać tylko wspólne fakty! Got: {centroid_facts}"


def test_centroid_method_specialized():






    rules = [
        Rule(1, [Fact("a", "1"), Fact("b", "2"), Fact("c", "3")], Fact("dec", "A")),
        Rule(2, [Fact("a", "1"), Fact("b", "2"), Fact("d", "4")], Fact("dec", "A")),
        Rule(3, [Fact("a", "1"), Fact("e", "5")], Fact("dec", "A")),
    ]

    clusterer = RuleClusterer(n_clusters=1, centroid_method='specialized')
    centroid = clusterer._compute_centroid(rules, cluster_id=0)

    centroid_facts = set((p.attribute, p.value) for p in centroid.premises)


    expected = {("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"), ("e", "5")}
    assert centroid_facts == expected, \
        f"Profile_specialized powinien zawierać wszystkie fakty! Got: {centroid_facts}"


def test_centroid_method_weighted():






    rules = [
        Rule(1, [Fact("a", "1"), Fact("b", "2"), Fact("c", "3")], Fact("dec", "A")),
        Rule(2, [Fact("a", "1"), Fact("b", "2"), Fact("d", "4")], Fact("dec", "A")),
        Rule(3, [Fact("a", "1"), Fact("e", "5")], Fact("dec", "A")),
    ]




    clusterer = RuleClusterer(n_clusters=1, centroid_method='weighted', centroid_threshold=0.5)
    centroid = clusterer._compute_centroid(rules, cluster_id=0)

    centroid_facts = set((p.attribute, p.value) for p in centroid.premises)


    assert ("a", "1") in centroid_facts, "a=1 (100%) musi być w centroidzie!"
    assert ("b", "2") in centroid_facts, "b=2 (67%) musi być w centroidzie!"

    assert ("c", "3") not in centroid_facts, "c=3 (33%) nie powinien być w centroidzie!"
    assert ("d", "4") not in centroid_facts, "d=4 (33%) nie powinien być w centroidzie!"
    assert ("e", "5") not in centroid_facts, "e=5 (33%) nie powinien być w centroidzie!"


def test_centroid_method_weighted_high_threshold():



    rules = [
        Rule(1, [Fact("a", "1"), Fact("b", "2"), Fact("c", "3")], Fact("dec", "A")),
        Rule(2, [Fact("a", "1"), Fact("b", "2"), Fact("d", "4")], Fact("dec", "A")),
        Rule(3, [Fact("a", "1"), Fact("e", "5")], Fact("dec", "A")),
    ]



    clusterer = RuleClusterer(n_clusters=1, centroid_method='weighted', centroid_threshold=0.9)
    centroid = clusterer._compute_centroid(rules, cluster_id=0)

    centroid_facts = set((p.attribute, p.value) for p in centroid.premises)


    assert ("a", "1") in centroid_facts, "a=1 (100%) musi być w centroidzie!"

    assert ("b", "2") not in centroid_facts, "b=2 (67%) nie powinien przejść threshold 90%!"


def test_centroid_general_empty_intersection_fallback():





    rules = [
        Rule(1, [Fact("a", "1")], Fact("dec", "A")),
        Rule(2, [Fact("b", "2")], Fact("dec", "A")),
        Rule(3, [Fact("c", "3")], Fact("dec", "A")),
    ]

    clusterer = RuleClusterer(n_clusters=1, centroid_method='general')
    centroid = clusterer._compute_centroid(rules, cluster_id=0)



    assert len(centroid.premises) == 1, \
        f"Profile_general bez wspólnych przesłanek powinien mieć 1 fallback premise! Got: {len(centroid.premises)}"


def test_all_centroid_methods_produce_valid_rules():



    rules = [
        Rule(1, [Fact("a", "1"), Fact("b", "2")], Fact("dec", "A")),
        Rule(2, [Fact("a", "1"), Fact("c", "3")], Fact("dec", "A")),
    ]

    for method in ['general', 'specialized', 'weighted']:
        clusterer = RuleClusterer(n_clusters=1, centroid_method=method)
        centroid = clusterer._compute_centroid(rules, cluster_id=0)


        assert isinstance(centroid, Rule), f"Method {method}: centroid musi być instancją Rule!"

        assert centroid.conclusion == Fact("dec", "A"), f"Method {method}: nieprawidłowa konkluzja!"

        assert centroid.id >= 1_000_000, f"Method {method}: ID centroidu musi być >= 1_000_000!"
