import sys

# sys.path.append("../entitylinking")
from core import entity_linker
from datasets import dataset
from core.sentence import Sentence

import config_utils

config, logger = config_utils.load_config("../entitylinking/default_config.yaml")


def test_load_webqsp():
    linking_config = config['entity.linking']
    linker = entity_linker.BaseLinker(logger=logger, **linking_config['linker.options'])

    dataset_config = config['dataset']
    dataset_config['path_to_dataset'] = "../data/WebQSP/preprocessed/webqsp.train.entities.json"

    logger.info("Load the data, {} from {}".format(dataset_config['type'], dataset_config['path_to_dataset']))
    evaluation_dataset = dataset.WebQSPDataset(logger=logger, **dataset_config)
    samples = evaluation_dataset.get_samples()

    print(samples[0])

    sentence = Sentence(input_text=samples[0][1])
    linker.link_entities_in_sentence_obj(sentence)
    entities = [(l.get('kbID'),) + tuple(e['offsets']) for e in sentence.entities
                for l in e['linkings']
                if len(e['linkings']) > 0]
    print(entities)


if __name__ == '__main__':
    test_load_webqsp()

