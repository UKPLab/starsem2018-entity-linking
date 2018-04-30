from typing import Dict

import click
import datetime
import numpy as np

from entitylinking import config_utils
from entitylinking.core import entity_linker
from entitylinking.core.sentence import Sentence, load_candidates
from entitylinking.core.ml_entity_linker import MLLinker
from entitylinking.datasets import dataset
from entitylinking.evaluation import measures
from entitylinking.mlearning import models


@click.command()
@click.argument('config_file_path')
@click.argument('seed', default=-1)
@click.argument('gpuid', default=-1)
def train(config_file_path, seed, gpuid):

    config, logger = config_utils.load_config(config_file_path, seed, gpuid)

    linking_config = config['entity.linking']
    dataset_config = config['dataset']

    # Instantiate the linker
    logger.info("Load candidates.")
    precomputed_candidates = load_candidates(linking_config['linker.options']['precomputed_candidates'])
    linking_config['linker.options']['precomputed_candidates'] = precomputed_candidates

    logger.info("Load the data, {} from {}".format(dataset_config['type'], dataset_config['path_to_dataset']))
    evaluation_dataset = getattr(dataset, dataset_config['type'])(logger=logger, **dataset_config)

    training_samples = evaluation_dataset.get_samples()
    validation_samples, validation_targets = None, None
    if config['training'].get('train.on.full'):
        training_samples += evaluation_dataset.get_samples(dev=True)
    else:
        validation_samples = pack_data(evaluation_dataset.get_samples(dev=True),
                                       precomputed_candidates)
        validation_targets = np.asarray(validation_samples[-1])

    training_samples = pack_data(training_samples,
                                 precomputed_candidates,
                                 negatives_per_entity=config['training'].get('negative.samples.per.entity', -1),
                                 null_per_dataset=config['training'].get('null.per.dataset', -1))
    logger.info('Number of negative entities: train {}, val {}'.format(np.sum(np.asarray(training_samples[-1]) == 0),
                                                                       np.sum(validation_targets == 0) if validation_targets is not None else 0))

    if config['training']['optimize.parameters']:
        logger.info("** Optimize model parameters with random search. ** ")
        from entitylinking.mlearning.training import optimize
        optimize.optimize(training_config=config['training'], model_config=config['model'],
                          train_data=training_samples, dev_data=validation_samples,
                          eval_dataset=evaluation_dataset, logger=logger)
    else:
        trainablemodel = getattr(models, config['training'].get('model.type', "VectorModel"))(parameters=config['model'], logger=logger)
        trainablemodel.prepare_model()
        if validation_samples:
            validation_samples = (trainablemodel.encode_batch(validation_samples[:-1]), validation_samples[-1])
        training_samples = (trainablemodel.encode_batch(training_samples[:-1]), training_samples[-1])
        trainablemodel.train(training_samples, dev=validation_samples,
                             eval_on_dataset=lambda: evaluation_dataset.eval(MLLinker(model=trainablemodel,
                                                                                      **linking_config['linker.options']),
                                                                             verbose=False))
        print('Training finished')
        if config['model']['model.checkpoint']:
            trainablemodel.load_last_saved()

        mllinker = MLLinker(model=trainablemodel, logger=logger, **linking_config['linker.options'])
        print("Evaluate with dataset, micro-average")
        results = evaluation_dataset.eval(mllinker)
        print("Results: {}".format(results))

        main_entity_results = results
        print("Evaluate with dataset (main entity), micro-average")
        if any(len(t[3]) > 0 for t in evaluation_dataset.get_samples(dev=True)):
            mllinker._one_entity_mode = True
            main_entity_results = evaluation_dataset.eval(mllinker, only_the_main_entity=True)
            print("Results: {}".format(main_entity_results))
            mllinker._one_entity_mode = False

        print(trainablemodel._save_model_to + trainablemodel._model_file_name)
        # print('Evaluate instance-based')
        # baseline = np.ones_like(validation_samples[1], dtype=np.int8)
        # b_prec, b_rec, b_f1 = measures.prec_rec_f1(baseline, validation_targets, empty_guessed=0)
        # accuracy = len((baseline == validation_targets).nonzero()[0]) / len(validation_targets)
        # print("Baseline: ", accuracy, b_prec, b_rec, b_f1)
        # predictions_instance_base = trainablemodel.scores_for_instance(validation_samples[0])
        # print("Prediction for the instances: ", predictions_instance_base.shape)
        # predictions_instance_base = np.argmax(predictions_instance_base, axis=-1)
        # accuracy = np.sum(predictions_instance_base == validation_targets) / len(validation_targets)
        # b_prec, b_rec, b_f1 = measures.prec_rec_f1(predictions_instance_base, validation_samples[-1], empty_guessed=0)
        # print("Model predictions: ", accuracy, b_prec, b_rec, b_f1)
        # negative_gold = validation_targets == 0
        # print("Recall (negatives): ",  np.sum(predictions_instance_base[negative_gold] == validation_targets[negative_gold])
        #       / np.sum(negative_gold))


def pack_data(annotation_samples, precomputed_candidates,
              negatives_per_entity=-1, null_per_dataset=-1):
    entities_data, candidates_data, targets = [], [], []
    nulls_counter = 0
    broadest_linker_settings = {"max_mention_len": 4, "num_candidates": 100,
                                "prefer_longer_matches": False, "no_mentions_overlap": False,
                                "caseless_mode": True, "mention_context_size": 2}
    entitylinker = entity_linker.HeuristicsLinker(**broadest_linker_settings,
                                                  precomputed_candidates=precomputed_candidates)
    for el_id, text, annotations, _, _ in annotation_samples:
        annotations = {e[0] for e in annotations}
        sentence = precomputed_candidates[el_id]
        sentence = entitylinker.link_entities_in_sentence_obj(sentence)
        for entity in sentence.entities:
            candidate_linkings = entity.get("linkings", [])
            negatives_counter = 0
            current_candidates = []
            target = 0
            entity_copy = {k: v for k, v in entity.items() if k != "linkings"}
            for i, l in enumerate(candidate_linkings):
                l['retrieval.rang'] = i
                is_gold = l.get("kbID") in annotations and target == 0
                if not is_gold:
                    negatives_counter += 1
                elif target == 0:
                    target = len(current_candidates) + 1
                if is_gold or negatives_per_entity < 0 or negatives_counter <= negatives_per_entity:
                    current_candidates.append(l)
            if target == 0:
                nulls_counter += 1
            if null_per_dataset < 0 or target != 0 or nulls_counter <= null_per_dataset:
                entities_data.append(entity_copy)
                targets.append(target)
                candidates_data.append(current_candidates)
    assert len(entities_data) == len(candidates_data) == len(targets)
    return entities_data, candidates_data, targets


if __name__ == "__main__":
    train()
