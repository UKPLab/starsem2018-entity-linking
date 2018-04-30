import datetime
import click

from entitylinking import config_utils
from entitylinking import core
from entitylinking.datasets import dataset


@click.command()
@click.argument('path_to_model')
@click.argument('config_file_path')
def evaluate(path_to_model, config_file_path):

    config, logger = config_utils.load_config(config_file_path)

    linking_config = config['entity.linking']
    dataset_config = config['dataset']
    # Instantiate the linker
    logger.info("Instantiate a {} linker".format(linking_config['linker']))
    linking_config['linker.options']['path_to_model'] = path_to_model
    entitylinker = getattr(core, linking_config['linker'])(logger=logger, **linking_config['linker.options'])

    logger.info("Load the data, {} from {}".format(dataset_config['type'], dataset_config['path_to_dataset']))
    evaluation_dataset = getattr(dataset, dataset_config['type'])(logger=logger, **dataset_config)

    logger.info("Evaluate")
    fb = 'fb' in config['evaluation'] and config['evaluation']['fb']
    results = evaluation_dataset.eval(entitylinker, fb=fb)
    print("Results: {}".format(results))
    logger.info("Evaluate (Main only)")
    main_entity_results = results
    if any(len(t[3]) > 0 for t in evaluation_dataset.get_samples(dev=True)):
        entitylinker._one_entity_mode = True
        main_entity_results = evaluation_dataset.eval(entitylinker, only_the_main_entity=True, fb=fb)
        print("Results: {}".format(main_entity_results))

    now = datetime.datetime.now()
    with open(f"results/seed_time_{now.date()}_{now.microsecond}.txt", "w") as out:
        out.write(dataset_config['path_to_dataset'] + "\n")
        out.write(path_to_model + "\n")
        out.write("{}\n".format(results))
        out.write("{}\n".format(main_entity_results))


if __name__ == "__main__":
    evaluate()
