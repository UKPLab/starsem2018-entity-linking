import abc
from collections import defaultdict
import json

import tqdm

from entitylinking.base_objects import Loggable
from entitylinking.core.entity_linker import BaseLinker
from entitylinking.core.sentence import Sentence
from entitylinking.evaluation import measures
from entitylinking.wikidata import queries


class Dataset(Loggable, metaclass=abc.ABCMeta):

    def __init__(self,
                 subsample=1.0,
                 **kwargs):
        super(Dataset, self).__init__(**kwargs)
        self._subsample_ratio = subsample

    @abc.abstractmethod
    def get_samples(self, dev=False, fb=False):
        raise NotImplementedError

    def _subsample(self, samples):
        if self._subsample_ratio < 1.0:
            samples = samples[:int(len(samples) * self._subsample_ratio)]
        return samples

    def precompute(self, linker: BaseLinker, verbose=True):
        """
        Extract entities from each dataset instance and compute linking candidates with a BaseLinker.

        :param linker: an instance of a BaseLinker
        :param verbose: if True, progress is indicated
        :return: a dictionary that maps instance ids to candidate annotations
        """
        gold_total = 0
        predicted_correct = 0

        data_iterator = tqdm.tqdm(self.get_samples() + self.get_samples(dev=True), ncols=100, ascii=True, disable=not verbose)
        precomputed_candidates = {}

        for el_id, text, annotations, _ in data_iterator:
            sentence = Sentence(input_text=text)
            sentence = linker.link_entities_in_sentence_obj(sentence)
            entities = [(l.get('kbID'),) + tuple(e['offsets']) for e in sentence.entities
                        for l in e['linkings']
                        if len(e['linkings']) > 0]

            match = measures.entity_linking_tp_with_overlap(annotations, entities)

            predicted_correct += match
            gold_total += len(annotations)
            recall = predicted_correct / gold_total if gold_total > 0 else 0
            data_iterator.set_postfix(rec=recall)
            precomputed_candidates[el_id] = sentence

        return precomputed_candidates

    def eval(self, linker: BaseLinker,
             only_the_main_entity=False,
             fb=False,
             verbose=True):
        performance_per_entity_type = defaultdict(lambda: [0, 0, 0])
        predicted_correct = 0
        predicted_total = 0
        gold_total = 0

        data_iterator = tqdm.tqdm(self.get_samples(dev=True, fb=fb), ncols=100, ascii=True, disable=not verbose)

        for el_id, text, annotations, main_entity, gold_entity_classes in data_iterator:
            sentence = Sentence(input_text=text)
            sentence = linker.link_entities_in_sentence_obj(sentence, element_id=el_id)
            entities = [(l.get('kbID'),) + tuple(e['offsets']) for e in sentence.entities
                        for l in e['linkings']
                        if len(e['linkings']) > 0]
            entity_classes = [queries.get_mapped_entity_type(e[0]) if e else "other" for e in entities]
            if fb:
                entities = [(l.get('fbID'),) + tuple(e['offsets']) for e in sentence.entities
                        for l in e['linkings']
                        if len(e['linkings']) > 0]
            if only_the_main_entity:
                annotations = [main_entity]
                match = measures.entity_linking_tp_with_overlap(annotations, entities)
            else:
                entities = [e[0] for e in entities]
                annotations = [e[0] for e in annotations]
                match = 0
                for ai, a in enumerate(annotations):
                    gold_entity_class = gold_entity_classes[ai] if gold_entity_classes and gold_entity_classes[ai] else "other"
                    if a in entities:
                        match += 1
                        performance_per_entity_type[gold_entity_class][0] += 1
                    performance_per_entity_type[gold_entity_class][2] += 1
                for entity_class in entity_classes:
                    performance_per_entity_type[entity_class][1] += 1


            predicted_correct += match
            predicted_total += len(entities)
            gold_total += len(annotations)
            precision = predicted_correct / predicted_total if predicted_total > 0 else 0
            recall = predicted_correct / gold_total if gold_total > 0 else 0
            f1 = (2.0*precision*recall) / (precision + recall) if precision + recall > 0 else 0
            data_iterator.set_postfix(prec=precision,
                                      rec=recall,
                                      f1=f1)

        precision = predicted_correct / predicted_total if predicted_total > 0 else 0
        recall = predicted_correct / gold_total if gold_total > 0 else 0
        f1 = (2.0*precision*recall)/(precision + recall) if precision + recall > 0 else 0

        for cls, stats in performance_per_entity_type.items():
            predicted_correct, predicted_total, gold_total = tuple(stats)
            cls_precision = predicted_correct / predicted_total if predicted_total > 0 else 0
            cls_recall = predicted_correct / gold_total if gold_total > 0 else 0
            cls_f1 = (2.0*cls_precision*cls_recall) / (cls_precision + cls_recall) if cls_precision + cls_recall > 0 else 0
            performance_per_entity_type[cls] = (cls_precision, cls_recall, cls_f1)

        return precision, recall, f1, dict(performance_per_entity_type)


class WebQSPDataset(Dataset):

    def __init__(self,
                 path_to_dataset,
                 debug_mode=False,
                 **kwargs):
        """
        WebQuestions entity linking dataset.

        :param path_to_dataset: path to the preprocessed dataset in json format.
        >>> WebQSPDataset(path_to_dataset="../../data/WebQSP/preprocessed/webqsp.train.entities.json") is not None
        True
        """
        super(WebQSPDataset, self).__init__(**kwargs)
        self._questions = {}
        with open(path_to_dataset) as f:
            self._questions = json.load(f)
            self._questions = {q_obj['question_id']: q_obj for q_obj in self._questions}
        self._train_ids = []
        self._dev_ids = []
        if ".train." in path_to_dataset:
            with open(path_to_dataset.replace("train.entities.", "dev.ids.")) as f:
                self._dev_ids = json.load(f)
            if debug_mode:
                self._train_ids = self._dev_ids
            else:
                with open(path_to_dataset.replace("train.entities.", "train.ids.")) as f:
                    self._train_ids = json.load(f)
        else:
            self._dev_ids = list(self._questions.keys())

    def get_samples(self, dev=False, fb=False):
        questions = [self._questions[i] for i in (self._dev_ids if dev else self._train_ids)]
        samples = [(q_obj['question_id'],
                    q_obj['utterance'],
                    [(e,) for e in q_obj['entities_fb' if fb else 'entities']],
                    ((q_obj['main_entity_fb' if fb else 'main_entity'],) + tuple(q_obj['main_entity_pos'])) if 'main_entity' in q_obj else (),
                    q_obj.get('entity_classes')
                    )
                   for q_obj in questions]
        if not dev:
            samples = self._subsample(samples)
        return samples


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
