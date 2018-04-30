import urllib
import json

from entitylinking.wikidata import queries
from entitylinking.core.entity_linker import BaseLinker
from entitylinking.core.sentence import Sentence


class DBPediaLinker(BaseLinker):

    def __init__(self,
                 confidence=0.5,
                 **kwargs):
        super(DBPediaLinker, self).__init__(**kwargs)
        self._spotlight_url = "http://model.dbpedia-spotlight.org/en/annotate?"
        self._confidence = confidence

    def compute_candidate_scores(self, entity, tagged_text):
        return entity

    def link_entities_in_sentence_obj(self, sentence_obj: Sentence, element_id=None, num_candidates=-1):
        sentence_obj = Sentence(input_text=sentence_obj.input_text, tagged=sentence_obj.tagged,
                                mentions=sentence_obj.mentions, entities=sentence_obj.entities)
        sentence_obj.entities = []

        params = urllib.parse.urlencode({'text': sentence_obj.input_text, 'confidence': str(self._confidence)})
        request = urllib.request.Request(self._spotlight_url + params)
        request.add_header("Accept", "application/json")
        try:
            content = json.loads(urllib.request.urlopen(request).read())
            sentence_obj.entities = [{
                                        "linkings": [{
                                                        'kbID': queries.map_wikipedia_id(r.get("@URI")
                                                                                                  .replace("http://dbpedia.org/resource/", "")
                                                                                                  .replace("http://dbpedia.org/page/", ""))
                                                      }],
                                        'offsets': (int(r.get('@offset', '0')), int(r.get('@offset', '0')) + len(r.get('@surfaceForm', "")))}
                                     for r in content.get('Resources', [])]
        except:
            pass
        return sentence_obj
