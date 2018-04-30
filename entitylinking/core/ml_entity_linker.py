from entitylinking.core.entity_linker import HeuristicsLinker
from entitylinking.mlearning import models


class MLLinker(HeuristicsLinker):

    def __init__(self,
                 model=None,
                 path_to_model=None,
                 drop_entities=True,
                 confidence=0.5,
                 **kwargs):
        super(MLLinker, self).__init__(**kwargs)
        self._elmodel = None
        if model:
            self._elmodel = model
        elif path_to_model:
            model_type = path_to_model.split("/")[-1].split("_")[0]
            self._elmodel = getattr(models, model_type)(parameters={"models.save.path": "../trainedmodels/"}, logger=self.logger)
            self._elmodel.load_from_file(path_to_model)
        else:
            self.logger.error("Can't initialize the model")
        self._drop_entities = drop_entities
        self._confidence = confidence

    def compute_candidate_scores(self, entity, tagged_text):
        entity = super(MLLinker, self).compute_candidate_scores(entity, tagged_text)
        linkings = entity['linkings']
        if self._elmodel and len(linkings) > 0:
            data_encoded = self._elmodel.encode_batch(([entity], [linkings]))
            ml_scores = self._elmodel.scores_for_instance(data_encoded)[0]
            assert len(ml_scores) == len(linkings) + 1
            for s, l in zip(ml_scores[1:], linkings):
                l['ml_score'] = s
                l['score'] = (-l['ml_score'],) + l['score'][1:]
            entity['drop_score'] = ml_scores[0]
            if self._drop_entities and entity['drop_score'] > (1 - self._confidence):
                entity['linkings'] = []
                return entity

            entity['linkings'] = sorted(linkings, key=lambda l: (l['score']))
        return entity
