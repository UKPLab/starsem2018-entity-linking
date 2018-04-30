import logging
import string
from typing import List, Iterable

import numpy as np

from entitylinking import utils
from entitylinking.wikidata import wdaccess, wdscheme, queries

entity_linking_p = {
    "candidates.to.retrieve": 100,
    "max.match.diff": 2,
    "extended.candidate.features": True  # If True, retrieves semantic signatures and other additional heavy features
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Unfortunately there are label errors and typos in Wikidata, put the entity in the blacklist to block it
entity_blacklist = utils.load_list(utils.RESOURCES_FOLDER + "entity_blacklist.txt")
punctuation = set(string.punctuation)
entity_freq_map = utils.load_entity_freq_map(utils.RESOURCES_FOLDER + "wikidata_entity_freqs.map")
max_entity_freq = max(v for v in entity_freq_map.values())


def link_mention(entity_tokens: List, entity_tags: Iterable = None) -> List:
    """
    Link the given list of tokens to an entity in a knowledge base.
    The list is sorted by candidate entity frequency estimated on Wikipedia.

    :param entity_tokens: a list of tokens
    :param entity_tags: optional, part of speech tags for the entity tokens
    :return: a list of linkings as dictionaries where the "kbID" field contains the entity id
    >>> link_mention(['Martin', 'Luther', 'King', 'Junior'])
    [[('Q8027', 'Martin Luther King, Jr.'), ('Q6776048', 'Martin Luther King, Jr.')]]
    >>> link_mention(['movie'])
    [[('Q11424', 'film'), ('Q1179487', 'Movies'), ('Q6926907', 'Movies')]]
    >>> link_mention(['lord', 'of', 'the', 'rings'])
    [[('Q15228', 'The Lord of the Rings'), ('Q127367', 'The Lord of the Rings: The Fellowship of the Ring'), ('Q131074', 'The Lord of the Rings')]]
    >>> link_mention(['doin', 'me', ','])
    []
    >>> link_mention(['#justinbieber'])
    []
    """
    entity_tokens = [t for t in entity_tokens if t not in punctuation]
    if all(e.lower() in utils.stop_words_en for e in entity_tokens):
        return []

    linkings = wdaccess.query_wikidata(queries.query_get_entity_by_label(entity_tokens))
    if not linkings and entity_tags and all(t.startswith("NN") for t in entity_tags):
        entity_lemmas = utils.lemmatize_tokens(entity_tokens)
        if [l.lower() for l in entity_lemmas] != [t.lower() for t in entity_tokens]:
            linkings += wdaccess.query_wikidata(queries.query_get_entity_by_label(entity_lemmas))

    linkings = _post_process_entity_linkings(linkings, entity_tokens)
    return linkings


def _post_process_entity_linkings(linkings, mention_tokens):
    """
    Extract and compute features for each retrieved candidate linking

    :param linkings: possible linkings as a list of dictionaries
    :param mention_tokens: list of entity tokens as appear in the sentence
    :return: sorted linkings
import wikidata.queries    >>> sorted(_post_process_entity_linkings( \
            wdaccess.query_wikidata(wikidata.queries.entity_query(['movie']), starts_with=None), ['movies']), \
            key=lambda l: l['freq'], reverse=True)[0]['kbID']
    'Q11424'
    >>> _post_process_entity_linkings([{'anylabel': "I'm Doin' Me", 'id_rank': 15.60162328095359, 'label': "I'm Doin' Me"}, \
    {'anylabel': "Somebody's Doin' Me Right", 'kbID': 'Q7559520', 'label': "Somebody's Doin' Me Right"}], mention_tokens=['doin', 'me'])
    []
    """
    # Candidate linkings are returned as dictionaries, first we reformat them
    linkings = [l for l in linkings if 'e2' in l]
    if len(linkings) == 0:
        return linkings

    for l in linkings:
        l['kbID'] = l["e2"].replace(wdscheme.WIKIDATA_ENTITY_PREFIX, "")
        l['matchedlabel'] = l.get("matchedlabel", "")
        del l['e2']
        # To tokenize property titles, titles are usually 2-3 tokens and this is much faster than calling CoreNLP
        match_tokens = [t for t in utils.split_pattern.split(l.get("matchedlabel", "")) if t]
        l['match_tokens'] = match_tokens
        l['match_diff'] = len(match_tokens) - len(mention_tokens)

    # Since we are defaulting to a full text search, we have to remove matches that a much longer than the mention
    linkings = [l for l in linkings if l.get("kbID") not in entity_blacklist  # The blacklist to filter out KB errors
                and l.get("kbID", "").startswith("Q")  # To guard against accidentally retrieved KB support entities
                and l['match_diff'] <= entity_linking_p.get("max.match.diff", 2)]  # Compare the match length

    # Second, compute the edit distance between the main label of a candidate and the mention, retrieve entity frequency
    for l in linkings:
        l['mention_tokens_len'] = len(mention_tokens)
        l['freq'] = entity_freq_map.get(l['kbID'], 0)
        q_index = int(l['kbID'][1:])
        l['id_rank'] = np.log(q_index).item()

    # Sort candidates by entity frequency and do a cut-off
    linkings = sorted(linkings, key=lambda l: (l['match_diff'], len(l['matchedlabel']), -l['freq'], l['id_rank']))
    linkings = linkings[:entity_linking_p['candidates.to.retrieve']]

    # Extract additional computation-heavy features (for now ony the signature)
    if entity_linking_p["extended.candidate.features"]:
        for l in linkings:
            related_entities, related_relations = queries.get_semantic_signature(l['kbID'])
            related_entities = {r for r in related_entities if r[2] in wdscheme.frequent_properties and int(r[1][1:]) < 3000000}
            related_relations = {r for r in related_relations if r[1] in wdscheme.frequent_properties}
            sem_signature = {t.lower() for el in related_entities | related_relations for t in utils.split_pattern.split(el[0])}\
                            - utils.stop_words_en - {w.lower() for w in mention_tokens}
            l['signature'] = list(sem_signature)
            l['related_entities'] = list(related_entities)
            l['num_related_entities'] = len(related_entities)
            l['related_relations'] = list(related_relations)
            l['num_related_relations'] = len(related_relations)
    # Remove candidates that have empty semantic signatures
    linkings = [l for l in linkings if l.get('num_related_relations', 0) > 0]
    linkings = sorted(linkings, key=lambda l: l.get('num_related_relations', 0), reverse=True)

    # the method doesn't compute explicit scores
    return linkings

