import sys
import yaml
import logging

import numpy as np
import torch

from entitylinking.wikidata import wdaccess
from entitylinking.core import candidate_retrieval


def load_config(config_file_path, seed=-1, gpuid=-1):
    """
    Read config and set up the logger

    :param config_file_path: path to the config file in the yaml format
    :return: config, logger
    """
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file.read())
    config_global = config.get('global', {})

    logger = logging.getLogger(__name__)
    logger.setLevel(config['logger']['level'])
    ch = logging.StreamHandler()
    ch.setLevel(config['logger']['level'])
    logger.addHandler(ch)

    if seed < 0:
        np.random.seed(config_global.get('random.seed', 1))
        torch.manual_seed(config_global.get('random.seed', 1))
        logger.info("Seed: {}".format(config_global.get('random.seed', 1)))
    else:
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info("Seed: {}".format(seed))

    logger.debug(config)
    if "entity.linking" not in config:
        logger.error("Entity linking parameters not in the config file!")
        sys.exit()

    if torch.cuda.is_available():
        logger.info("Using your CUDA device")
        if seed < 0:
            torch.cuda.manual_seed(config_global.get('random.seed', 1))
        else:
            torch.cuda.manual_seed(seed)
        if gpuid < 0:
            torch.cuda.set_device(config_global.get('gpu.id', 0))
        else:
            torch.cuda.set_device(gpuid)
        logger.info("GPU ID: {}".format(torch.cuda.current_device()))

    wdaccess.set_backend(config['wikidata']['backend'])

    candidate_retrieval.entity_linking_p.update({k: config['entity.linking'][k] for k in config['entity.linking']
                                                 if k in candidate_retrieval.entity_linking_p})
    logger.debug(candidate_retrieval.entity_linking_p)

    return config, logger
