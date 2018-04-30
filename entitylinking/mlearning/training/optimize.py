import atexit
import hyperopt as hy
import json

from entitylinking.core import MLLinker
from entitylinking.mlearning import models
from entitylinking import utils

optimization_space = {
    'negative.class.weight': hy.hp.uniform('negative.class.weight', .1, .9),
    'negative.weight.epoch': hy.hp.quniform('negative.weight.epoch', 2, 20, 2),
    'sem.layer.size': hy.hp.choice('sem.layer.size', [2**x for x in range(4, 9)]),
    'neg.layer.size': hy.hp.choice('neg.layer.size', [2**x for x in range(4, 9)]),
    'poss.emb.size': hy.hp.choice('poss.emb.size', [3,5,7]),
    'enc.activation': hy.hp.choice('enc.activation', ['relu', 'tanh']),
    'enc.pooling': hy.hp.choice('enc.pooling', ['max', 'avg']),
    'dropout': hy.hp.uniform('dropout', .0, .5),
    'word.conv.size': hy.hp.choice('word.conv.size', [2**x for x in range(4, 9)]),
    'word.conv.width': hy.hp.choice('word.conv.width', [3,5]),
    'word.conv.depth': hy.hp.choice('word.conv.depth', [1,2,3]),
    'word.repeat.convs': hy.hp.choice('word.repeat.convs', [1,2,3]),
    'char.conv.size': hy.hp.choice('char.conv.size', [2**x for x in range(4, 9)]),
    'char.conv.width': hy.hp.choice('char.conv.width', [3,5]),
    'char.conv.depth': hy.hp.choice('char.conv.depth', [1,2,3]),
    'char.repeat.convs': hy.hp.choice('char.repeat.convs', [1,2,3]),
    'char.emb.size': hy.hp.choice('char.emb.size', [25*x for x in range(1,5)]),
    'entity.layer.size': hy.hp.choice('entity.layer.size', [25*x for x in range(1,5)]),
    'relation.layer.size': hy.hp.choice('relation.layer.size', [25*x for x in range(1,5)]),
}

trials_counter = 0
dev = None
train = None


def optimize(training_config, model_config, train_data, dev_data, eval_dataset, logger):
    trials = hy.Trials()
    atexit.register(lambda: wrap_up_optimization(trials, training_config['optimize.save.history'], logger))

    logger.debug("Loading embeddings")
    embedding_matrix, element2idx = utils.load_word_embeddings(model_config['word.embeddings'])
    entities_embedding_matrix, entity2idx, rels_embedding_matrix, rel2idx = utils.load_kb_embeddings(model_config['kb.embeddings'])

    def optimization_trial(sampled_parameters):
        global trials_counter, dev, train
        try:
            logger.info("** Trial: {}/{} ** ".format(trials_counter, training_config['optimize.num.trails']))
            trials_counter += 1
            sampled_parameters['negative.weight.epoch'] = int(sampled_parameters['negative.weight.epoch'])
            model_trial = getattr(models, training_config.get('model.type', "VectorModel"))(parameters={**model_config, **sampled_parameters}, logger=logger)
            model_trial.prepare_model(embedding_matrix=embedding_matrix, element2idx=element2idx,
                                      entities_embedding_matrix=entities_embedding_matrix, entity2idx=entity2idx,
                                      rels_embedding_matrix=rels_embedding_matrix, rel2idx=rel2idx)
            if train is None and dev is None:
                dev = (model_trial.encode_batch(dev_data[:-1]), dev_data[-1])
                train = (model_trial.encode_batch(train_data[:-1]), train_data[-1])

            results = model_trial.train(train, dev=dev,
                                        eval_on_dataset=lambda: eval_dataset.eval(MLLinker(model=model_trial, logger=logger), verbose=False))
            results['actual_loss'] = results['v_loss']
            results['loss'] = 1.0 - results['v_f1']
            return {**results, 'status': hy.STATUS_OK, 'sampled.parameters': sampled_parameters}
        except Exception as ex:
            logger.error(ex)
            return {'loss': -1, 'status': hy.STATUS_FAIL, 'sampled.parameters': sampled_parameters}

    hy.fmin(optimization_trial,
            optimization_space,
            algo=hy.rand.suggest,
            max_evals=training_config['optimize.num.trails'],
            trials=trials, verbose=1)


def wrap_up_optimization(trials, save_to, logger):
    if len(trials.trials) > 0:
        logger.info("Optimization finished, best trail: {}".format(trials.best_trial))
        logger.info("Best parameters: {}".format(trials.best_trial['result']['sampled.parameters']))
        with open(save_to, 'w') as out:
            json.dump([(t['misc']['vals'], t['result']) for t in trials.trials], out)
