import json
import pickle


class Sentence:
    __slots__ = ['input_text', 'tagged', 'mentions', 'entities']

    def __init__(self, input_text=None, tagged=None, mentions=None, entities=None):
        """
        A sentence object.
        
        :param input_text: raw input text as a string
        :param tagged: a list of dict objects, one per token, with the output of the POS and NER taggers, see utils
                      for more info
        :param mentions: a list of dict object, one per mention, see mention_extraction for mre info
        :param entities: a list of tuples, where each tuple is an entity link (first position is the KB id and 
                         the second position is the label)
        """
        self.input_text = input_text
        self.tagged = tagged
        self.mentions = mentions
        self.entities = entities

    def __repr__(self):
        return "Sentence({})".format({s: getattr(self, s) for s in self.__slots__})


class SentenceEncoder(json.JSONEncoder):
    """
    >>> s = Sentence("the", [("the", "ART")])
    >>> json.dumps(s, cls=SentenceEncoder)
    '{"input_text": "the", "tagged": [["the", "ART"]], "mentions": null, "entities": null}'
    >>> s = '{"input_text": "the", "tagged": [["the", "ART"]], "mentions": null, "entities": null}'
    >>> dict_to_sentence(json.loads(s))
    Sentence({'input_text': 'the', 'tagged': [['the', 'ART']], 'mentions': None, 'entities': None})
    """

    def default(self, o):
        if isinstance(o, Sentence):
            return {s: getattr(o, s) for s in o.__slots__}
        return super(SentenceEncoder).default(o)


def dict_to_sentence(o):
    return Sentence(**o)


def load_candidates(precomputed_candidates_path):
    if precomputed_candidates_path.endswith(".json"):
        with open(precomputed_candidates_path) as f:
            precomputed_candidates = json.load(f)
    else:
        with open(precomputed_candidates_path, "rb") as f:
            precomputed_candidates = pickle.load(f)
    for k in precomputed_candidates:
        precomputed_candidates[k] = dict_to_sentence(precomputed_candidates[k])
    return precomputed_candidates
