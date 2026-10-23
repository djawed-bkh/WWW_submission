import numpy as np

from . import time_package as tp


temporal_granularity = "D"


def to_common_uri(uri_relation):
    namespace_common = "http://www.wikidata.org/prop/P"
    namespace_direct = "http://www.wikidata.org/prop/direct/P"
    if uri_relation[: len(namespace_direct)] == namespace_direct:
        return namespace_common + uri_relation[len(namespace_direct) :]
    return uri_relation


def processing_date_unknown_allowed(date_raw) -> np.datetime64 | None:
    if date_raw == "None":
        return None
    return np.datetime64(date_raw, temporal_granularity)


def find_multivaluation_temporal(entities):
    """
    Identifie les relations multi-valuées temporellement.
    """
    temporal_multivalued_relations = set()

    for entity in entities.values():
        for relation in entity.triples_per_p:
            time_sequence = tp.TimeSequence(
                tp.ordered_time_sequence_first_start(entity, relation)
            )
            if time_sequence.multi_valuation_temporal:
                temporal_multivalued_relations.add(relation)

    return temporal_multivalued_relations
