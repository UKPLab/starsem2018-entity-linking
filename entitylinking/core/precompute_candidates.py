import click

import json

from entitylinking import config_utils
from entitylinking.core import entity_linker
from entitylinking.core.sentence import SentenceEncoder
from entitylinking.datasets import dataset


@click.command()
@click.argument('config_file_path', default="../default_config.yaml")
def precompute(config_file_path):

    config, logger = config_utils.load_config(config_file_path)

    linking_config = config['entity.linking']
    dataset_config = config['dataset']
    # Instantiate the linker
    logger.info("Instantiate a BaseLinker linker")
    entitylinker = entity_linker.BaseLinker(logger=logger, **linking_config['linker.options'])

    logger.info("Load the data, {} from {}".format(dataset_config['type'], dataset_config['path_to_dataset']))
    evaluation_dataset = getattr(dataset, dataset_config['type'])(path_to_dataset=dataset_config['path_to_dataset'],
                                     subsample=dataset_config['subsample'],
                                     logger=logger)

    logger.info("Precompute candidate linkings")
    precomputed_candidates = evaluation_dataset.precompute(entitylinker)
    logger.info("Save the candidates")
    with open(dataset_config['precomputed_candidates'], "w") as out:
        json.dump(precomputed_candidates, out, cls=SentenceEncoder)


if __name__ == "__main__":
    precompute()
