from collections import defaultdict

from  entitylinking.wikidata import queries
from entitylinking.core.entity_linker import BaseLinker
from entitylinking.core.sentence import Sentence
from entitylinking import utils


class SMARTLinker(BaseLinker):

    def __init__(self,
                 path_to_predictions,
                 confidence=0.0,
                 **kwargs):
        super(SMARTLinker, self).__init__(**kwargs)
        self.predictions = defaultdict(lambda: defaultdict(list))
        with open(path_to_predictions) as f:
            predictions_list = [l.strip().split("\t") for l in f.readlines()]
            for entry in predictions_list:
                self.predictions[entry[0]][entry[1]].append(entry)
            for _, v in self.predictions.items():
                for k in v:
                    v[k] = sorted(v[k], reverse=True, key=lambda x: float(x[6]))
        confidence_scores = sorted([c[0][6] for v in self.predictions.values() for c in v.values()])
        index = min(len(confidence_scores) - 1, int(len(confidence_scores)*confidence))
        self._confidence = float(confidence_scores[index])

    def compute_candidate_scores(self, entity, tagged_text):
        return entity

    def link_entities_in_sentence_obj(self, sentence_obj: Sentence, element_id=None, num_candidates=-1):
        sentence_obj = Sentence(input_text=sentence_obj.input_text, tagged=sentence_obj.tagged,
                                mentions=sentence_obj.mentions, entities=sentence_obj.entities)
        if not sentence_obj.tagged:
            sentence_obj.tagged = utils.get_tagged_from_server(sentence_obj.input_text,
                                                               caseless=sentence_obj.input_text.islower())
        sentence_obj.entities = []
        if element_id:
            smart_predictions = [([p[4].replace("/", ".")[1:] for p in candidates if float(p[6]) > self._confidence],
                                  int(candidates[0][2]), int(candidates[0][2]) + int(candidates[0][3]))
                                 for e, candidates in self.predictions[element_id].items() if len(candidates) > 0]

            for c, s, e in smart_predictions:
                linkings = []
                for p in c:
                    kbID = queries.map_f_id(p)
                    linkings.append({'fbID': p, 'kbID': kbID, 'label': queries.get_main_entity_label(kbID) if kbID else None})
                sentence_obj.entities.append({"linkings": linkings,
                                      'offsets': (s, e),
                                      'type': 'NNP',
                                      'poss': [],
                                      'token_ids': _offets_to_token_ids(s, e, sentence_obj.tagged),
                                      'tokens': []})

        for e in sentence_obj.entities:
            # If there are many linking candidates we take the top N, since they are still ordered
            if num_candidates > 0:
                e['linkings'] = e['linkings'][:num_candidates]
            else:
                e['linkings'] = e['linkings'][:self.num_candidates]

        return sentence_obj


def _offets_to_token_ids(offset_start, offset_end, tagged):
    token_ids = []

    for t in tagged:
        if t['characterOffsetBegin'] >= offset_start and t['characterOffsetEnd'] <= offset_end:
            token_ids.append(t['index'] - 1)

    return token_ids
