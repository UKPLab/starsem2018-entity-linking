import re
from collections import deque, defaultdict
from typing import List, Iterable

from entitylinking.base_objects import Loggable
from entitylinking import utils
from entitylinking.core import candidate_retrieval, mention_extraction
from entitylinking.core.sentence import Sentence, load_candidates
from entitylinking.wikidata import wdscheme

content_word_pattern = re.compile(r"^[VNJ]")


class BaseLinker(Loggable):

    def __init__(self,
                 max_mention_len=4,
                 num_candidates=1,
                 prefer_longer_matches=True,
                 longer_matches_min_freq=1000,
                 no_mentions_overlap=True,
                 caseless_mode=False,
                 one_entity_mode=False,
                 only_frequent_entities=False,
                 precomputed_candidates=None,
                 **kwargs):
        """

        :param max_mention_len:
        :param num_candidates: 
        :param prefer_longer_matches:
        :param no_mentions_overlap:
        """
        super(BaseLinker, self).__init__(**kwargs)
        self.max_mention_len = max_mention_len
        self.num_candidates = num_candidates
        self.prefer_longer_matches = prefer_longer_matches
        self.longer_matches_min_freq = longer_matches_min_freq
        self.no_mentions_overlap = no_mentions_overlap
        self.caseless_mode = caseless_mode
        self.one_entity_mode = one_entity_mode
        self.only_frequent_entities = only_frequent_entities
        self._precomputed_candidates = None
        if isinstance(precomputed_candidates, dict):
            self._precomputed_candidates = precomputed_candidates
        elif precomputed_candidates is not None:
            self._precomputed_candidates = load_candidates(precomputed_candidates)

    def link_entities_in_sentence_obj(self, sentence_obj: Sentence, element_id=None, num_candidates=-1) -> Sentence:
        """
        The method takes a sentence dictionary object that might already contain a tagged input or recognized mentions.
        This is useful if tagging and mentioned extraction is done in bulk before the entity linking step.
        Supported fields in the sentence_obj object:
            "input_text": raw input text as a string
            "tagged": a list of dict objects, one per token, with the output of the POS and NER taggers, see utils
                      for more info (optional)
            "mentions": a list of dict object, one per mention, see mention_extraction for more info (optional)
            "entities": extracted entity candidates (optional)
        See Sentence for more info.
        
        :param sentence_obj: input sentence as a dictionary, might be an empty dict
        :param element_id: sentence id to retrieve precomputed candidates for certain linkers
        :param num_candidates: the number of candidate entity links to store for each entity. 
                                If set to more than 0 it will override the class setting.
        :return: the same sentence_obj object with a new field "entities"
        
        >>> l = HeuristicsLinker()
        >>> l.link_entities_in_sentence_obj(Sentence("Where does Norway get their oil?")).entities[0]['linkings']  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [('Q20', 'Norway'), ...]
        """
        if self._precomputed_candidates is not None and element_id in self._precomputed_candidates:
            sentence_obj = self._precomputed_candidates[element_id]
        sentence_obj = Sentence(input_text=sentence_obj.input_text, tagged=sentence_obj.tagged,
                                mentions=sentence_obj.mentions, entities=sentence_obj.entities)
        if sentence_obj.tagged is None and sentence_obj.input_text is not None:
            sentence_obj.tagged = utils.get_tagged_from_server(sentence_obj.input_text,
                                                               caseless=sentence_obj.input_text.islower())
            self.logger.debug([(t['word'], t['pos']) for t in sentence_obj.tagged])

        if sentence_obj.entities is None:
            if sentence_obj.mentions is not None:
                sentence_obj.entities = self._link_mentions_to_entities(sentence_obj.mentions)
            else:
                sentence_obj.entities = self._link_entities_in_tagged_input(sentence_obj.tagged)
                self.logger.debug([e['linkings'][0] for e in sentence_obj.entities])
        elif self.prefer_longer_matches:
            sentence_obj.entities = self._prefer_longer_matches(sentence_obj.entities)

        for e in sentence_obj.entities:
            e['text'] = sentence_obj.input_text
        sentence_obj.entities = [self.compute_candidate_scores(e, tagged_text=sentence_obj.tagged)
                                 for e in sentence_obj.entities]

        if self.no_mentions_overlap:
            if not self.one_entity_mode:
                sentence_obj.entities = resolve_entity_overlap_beam_search(sentence_obj.entities)
            else:
                sentence_obj.entities = sorted(sentence_obj.entities, key=lambda x: x.get('drop_score', 0.0))
                sentence_obj.entities = sentence_obj.entities[:1]

                # One mention span -> one entity. Each entity can have multiple linking candidates.
        for e in sentence_obj.entities:
            # If there are many linking candidates we take the top N, since they are still ordered
            if num_candidates > 0:
                e['linkings'] = e['linkings'][:num_candidates]
            else:
                e['linkings'] = e['linkings'][:self.num_candidates]

        return sentence_obj

    def link_entities_in_raw_input(self, input_text: str, element_id: str=None, num_candidates=-1) -> Sentence:
        """
        Takes a raw input string, extracts mentions and returns a list of the most probable entities that can be linked
         to the given input text.

        :param input_text: the input sentence as a string
        :param element_id: sentence id
        :param num_candidates: the number of candidate entity links to store for each entity.
                                If set to more than 0 it will override the class setting.
        :return: a list of tuples where the first element is the entity id and the second is the entity label
        >>> l = HeuristicsLinker(num_candidates=1)
        >>> l.link_entities_in_raw_input("Who wrote the song hotel California?")
        [('Q7366', 'song', (14, 18), [3]), ('Q780394', 'Hotel California', (19, 35), [4, 5])]
        >>> l.link_entities_in_raw_input("Donovan McNabb'strade to the Vikings is in place.")  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [('Q963185', 'Donovan McNabb', (0, 14), [0, 1]), ...]
        >>> l.link_entities_in_raw_input("what was the queen album?")
        [('Q15862', 'Queen', (13, 18), [3]), ('Q482994', 'album', (20, 24), [4])]
        """
        sentence = Sentence(input_text=input_text)
        sentence = self.link_entities_in_sentence_obj(sentence, element_id=element_id, num_candidates=num_candidates)
        sentence.entities = [{k: e[k] for k in {'type', 'linkings', 'token_ids', 'poss', 'tokens'}}
                             for e in sentence.entities
                             if len(e['linkings']) > 0]
        for e in sentence.entities:
            e['linkings'] = [(l.get('kbID'), l.get('label')) for l in e['linkings']]
        return sentence

    def _link_entities_in_tagged_input(self, tagged_text: List) -> List:
        """
        A submethod that first extracts mentions from the given tagged input and them tries to link each of them
        in turn to the KB by looking for candidate entity links that would match the mentions the best.
        
        The method returns a list of entities that is effectively a copy of the mentions but with added candidate 
        linkings.
        
        Use this method in no mentions are already annotated. The method extract shorter mentions at each step and
        tries to find a matching entity for the complete mention (i.e. that contains all mention tokens).
        
        :param tagged_text: same as the tagged attribute of the Sentence object.
        :return: a list of entities as dicts.
        """
        entities = []
        linked_token_ids = set()
        for mention_len in range(min(len(tagged_text), self.max_mention_len), 0, -1):
            mentions = mention_extraction.extract_entities(tagged_text, mention_len)
            mentions = [m for m in mentions if len(set(m['token_ids']) & linked_token_ids) == 0]
            self.logger.debug([f['tokens'] for f in mentions])
            if mentions:
                entities_ = self._link_mentions_to_entities(mentions)
                entities.extend(entities_)
                if self.prefer_longer_matches:
                    linked_token_ids |= {t_id for e in entities_ for t_id in e.get("token_ids", set())
                                         if any(l['freq'] > self.longer_matches_min_freq or l['match_diff'] == 0 for l in e['linkings'])}

        entities = sorted(entities, key=lambda e: e.get('token_ids', []))
        return entities

    def _prefer_longer_matches(self, entities):
        by_length = defaultdict(list)
        filtered_entities = []
        linked_token_ids = set()
        for e in entities:
            by_length[len(e['token_ids'])].append(e)

        for k in sorted(list(by_length.keys()), reverse=True):
            filtered = [m for m in by_length[k] if len(set(m['token_ids']) & linked_token_ids) == 0]
            filtered_entities.extend(filtered)
            linked_token_ids |= {t_id for e in filtered for t_id in e.get("token_ids", set())
                                 if any(l['freq'] > self.longer_matches_min_freq or l['match_diff'] == 0 for l in e['linkings'])}
        return filtered_entities

    def _link_mentions_to_entities(self, mentions: Iterable) -> Iterable:
        """
        For the given list of mentions, retrieve possible candidate linkings from the KB and sort them so that the most
        probably candidate comes first for each mention.
        
        The method returns a list of entities, where each entity corresponds to a single original mentions.
        
        :param mentions: a list of mentions as dicts with a 'tokens' field.
        :return: a list of entities as dicts.        
        """
        entities = deque()
        for mention in mentions:
            _linkings = candidate_retrieval.link_mention(mention['tokens'], entity_tags=mention.get('poss'))
            if _linkings is not None and len(_linkings) > 0:
                entity = {**mention, 'linkings': _linkings}
                entities.append(entity)
        return entities

    def compute_candidate_scores(self, entity, tagged_text):
        """
        The method should compute scores for each candidate linking for the givne entity and sort the candidates so that
        the most probable candidate comes first. 
        This is the core method of the class, each subclass (i.e. an implementation of an entity linker) should 
        implement it.
        
        :param entity: the current entity as an dict with a field 'linkings' that contains a list of candidate linkings 
        :param tagged_text: the current text as a list of tagged tokens 
        :return: the current entity with the 'linkings' field updated to include scores for each candidate
        """
        entity = {**entity}
        mention_label = " ".join(entity['tokens'])
        entity['linkings'] = [{**l} for l in entity['linkings']]
        entity['mention_chars'] = entity['text'][entity['offsets'][0]:entity['offsets'][1]]
        if self.only_frequent_entities:
            entity['linkings'] = [l for l in entity['linkings'] if l.get('freq', 0) > 0]
        for l in entity['linkings']:
            label = l.get("label", "")
            match_label = l.get("matchedlabel", "")
            if self.caseless_mode:
                label = label.lower()
                match_label = match_label.lower()
                mention_label = mention_label.lower()

            l['lev_main_label'] = utils.lev_distance(mention_label, label)
            l['lev_matchedlabel'] = utils.lev_distance(mention_label, match_label)
            l['num_related_relations'] = len(l.get('related_relations', []))

            l['score'] = (l['lev_matchedlabel'],
                          -l.get("freq"),
                          -l['num_related_relations'],
                          l['id_rank'])
        return entity


class HeuristicsLinker(BaseLinker):

    def __init__(self,
                 mention_context_size=2,
                 **kwargs):
        """
        The HeuristicsLinker computes a manually defined score based on the string features of the mention and relations
        of the candidate linkings to sort the candidates.
        
        :param mention_context_size: the size of the left and right context around the mention to use for comparison
        """
        super(HeuristicsLinker, self).__init__(**kwargs)
        self.mention_context_size = mention_context_size

    def compute_candidate_scores(self, entity, tagged_text):
        entity = super(HeuristicsLinker, self).compute_candidate_scores(entity, tagged_text)
        mention_tokens = entity['tokens']
        tokens = [t['word'] for t in tagged_text]
        min_t, max_t = min(entity['token_ids']), max(entity['token_ids'])
        mention_context = tokens[max(min_t - self.mention_context_size, 0):min_t] \
                           + mention_tokens \
                           + tokens[max_t + 1:max_t + 1 + self.mention_context_size]
        mention_context_str = " ".join(mention_context)
        entity['mention_context'] = mention_context_str
        entity['sentence_tokens'] = tokens
        entity['sentence_content_tokens'] = {t['word'].lower() for t in tagged_text if content_word_pattern.match(t['pos'])
                      and t['ner'] not in {"ORDINAL", "MONEY", "TIME", "PERCENTAGE"}} \
                     - utils.stop_words_en \
                     - {w.lower() for w in mention_tokens}
        linkings = entity['linkings']
        # linkings = [l for l in linkings if l.get("freq") > 0]
        for l in linkings:
            match_label = l.get("matchedlabel", "")
            if 'match_tokens' not in l:
                l['match_tokens'] = [t for t in utils.split_pattern.split(l.get("matchedlabel", "")) if t]
            l['label_tokens'] = [t for t in utils.split_pattern.split(l.get("label", "")) if t]

            if self.caseless_mode:
                match_label = match_label.lower()
                mention_context_str = mention_context_str.lower()
            l['lev_sentence'] = utils.lev_distance(mention_context_str, match_label)

            l['related_entities'] = [er for er in l['related_entities'] if er[2] in wdscheme.frequent_properties]
            l['singature_overlap'] = list(entity['sentence_content_tokens'] & set(l['signature']))
            l['singature_overlap_score'] = len(mention_tokens) + len(l['singature_overlap'])
            l['score'] = (
                -l['singature_overlap_score'],
                l['lev_sentence'] + l['lev_matchedlabel'],
                -l.get("freq"),
                -l['num_related_relations'],
                l['id_rank'])

        linkings = sorted(linkings, key=lambda l: l['score'])
        entity['linkings'] = linkings
        return entity


def resolve_entity_overlap(entities, single_entity_mode=False):
    """
    If there are overlapping entity links it will resolve them so that the best linkings are prefered.
    The method is greedy and doesn't guarantee the best solution.

    :param entities: a list of entities as dictionaries
    :param single_entity_mode: if True all entities are considered as overlapping and a single entity per list
    is selected
    :return:
    >>> resolve_entity_overlap([{'linkings': [{'anylabel': 'First Queen', 'label': 'First Queen', 'kbID': 'Q5453735', \
    'match_diff': 0, 'lev_main_label': 2, 'lev_matchedlabel': 2, 'lev_sentence': 8, 'id_rank': 15.511811253015074, \
    'score': (10, 25.511811253015075, 5453735), 'singature_overlap': [], 'singature_overlap_score': 10}, \
    {'anylabel': 'The first Queen', 'label': 'The first Queen', 'kbID': 'Q7776999', 'match_diff': 1, 'lev_main_label': 4,\
    'lev_matchedlabel': 4, 'lev_sentence': 10, 'id_rank': 15.866681089092785, 'score': (10, 29.866681089092786, 7776999), \
    'singature_overlap': [], 'singature_overlap_score': 10}], 'type': 'NNP', 'tokens': ['first', 'Queen'], 'token_ids': [0, 1]},\
    {'linkings': [{'anylabel': 'The Queen Album', 'label': 'The Queen Album', 'kbID': 'Q7758989', 'match_diff': 1, 'lev_main_label': 6,\
    'lev_matchedlabel': 6, 'lev_sentence': 10, 'id_rank': 15.864362600166613, 'score': (10, 31.864362600166615, 7758989), \
    'singature_overlap': [], 'singature_overlap_score': 10}, {'anylabel': 'On Air (Queen album)', 'label': 'On Air (Queen album)',\
    'kbID': 'Q27115596', 'match_diff': 2, 'lev_main_label': 9, 'lev_matchedlabel': 9, 'lev_sentence': 9, 'id_rank': 17.115619618469356,\
    'score': (10, 35.11561961846935, 27115596), 'singature_overlap': [], 'singature_overlap_score': 10}], 'type': 'NNP', \
    'tokens': ['Queen', 'album'], 'token_ids': [1, 2]}])[0]['token_ids']
    [0, 1]
    >>> resolve_entity_overlap([{'linkings':[{'score':1.0}], 'token_ids':[0,1,2]} ,{'linkings':[{'score':0.0}], 'token_ids':[1,2,3]}, \
     {'linkings':[{'score':2.0}], 'token_ids':[3,4]}, {'linkings':[{'score':1.0}], 'token_ids':[5,6]}])[-1]['token_ids']
    [5, 6]
    >>> resolve_entity_overlap([{'linkings':[{'score':(10, 1.0)}], 'token_ids':[3,4]}, {'linkings':[{'score':(10,2.0)}], 'token_ids':[3]}, {'linkings':[{'score':(9,0.0)}], 'token_ids':[4]}])[-1]['token_ids']
    [4]
    >>> resolve_entity_overlap([{'linkings':[{'score':1.0}], 'token_ids':[0,1,2]} ,{'linkings':[{'score':0.0}], 'token_ids':[1,2,3]}, \
     {'linkings':[{'score':2.0}], 'token_ids':[3,4]}, {'linkings':[{'score':1.0}], 'token_ids':[5,6]}], single_entity_mode=True)[-1]['token_ids']
    [1, 2, 3]
    """
    resolved_entities = []
    sorted_by_position = sorted([el for el in entities if el.get('token_ids')], key=lambda el: min(el['token_ids']))
    if len(sorted_by_position) == 0:
        return resolved_entities
    e = sorted_by_position.pop(0)
    while len(sorted_by_position) > 0:
        tokens = set(e.get('token_ids', []))
        e2 = sorted_by_position.pop(0)
        tokens_next = set(e2.get('token_ids', []))
        if len(tokens & tokens_next) > 0 or single_entity_mode:
            if 'linkings' in e and e['linkings'] and 'linkings' in e2 and e2['linkings']:
                e_first = e['linkings'][0]
                e2_first = e2['linkings'][0]
                if (e.get("drop_score", 0.0), e_first.get('score', ())) > \
                        (e2.get("drop_score", 0.0), e2_first.get('score', ())):
                    e = e2
        else:
            resolved_entities.append(e)
            e = e2
    resolved_entities.append(e)
    return resolved_entities


def resolve_entity_overlap_beam_search(entities):
    """
    If there are overlapping entity links it will resolve them so that the best linkings are prefered.
    The method is greedy and doesn't guarantee the best solution.

    :param entities: a list of entities as dictionaries
    :param single_entity_mode: if True all entities are considered as overlapping and a single entity per list
    is selected
    :return:
    >>> resolve_entity_overlap_beam_search([{'linkings':[{}], 'drop_score':0.7, 'token_ids':[0,1,2,3,4]}, \
                                              {'linkings':[{}], 'drop_score':0.6, 'token_ids':[1,2]}, \
                                              {'linkings':[{}], 'drop_score':0.78, 'token_ids':[3,4]}, \
                                              {'linkings':[{}], 'drop_score':0.8, 'token_ids':[5,6]}])

    """
    sorted_by_position = sorted([el for el in entities if el.get('token_ids')], key=lambda el: min(el['token_ids']))
    if len(sorted_by_position) == 0:
        return []
    e = sorted_by_position.pop(0)
    groups = [[e]]
    while len(sorted_by_position) > 0:
        e2 = sorted_by_position.pop(0)
        tokens_next = set(e2.get('token_ids', []))
        added = False
        for group in groups:
            tokens = set(group[-1].get('token_ids', []))
            if len(tokens & tokens_next) == 0:
                group.append(e2)
                added = True
        if not added:
            groups.append([e2])
    group_drop_scores = []
    for group in groups:
        group_drop_score = 1.0
        for e in group:
            group_drop_score *= e.get("drop_score", 0.0)
        group_drop_scores.append(group_drop_score)

    return_group = sorted(list(zip(groups, group_drop_scores)), key=lambda x: x[1])
    if return_group:
        return return_group[0][0]
    return []


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
