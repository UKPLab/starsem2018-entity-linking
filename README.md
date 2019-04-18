# Mixing Context Granularities for Improved Entity Linking on Question Answering Data across Entity Categories

## Entity linking with the Wikidata knowledge base

This is an accompanying repository for our ***SEM 2018 paper** ([pre-print](http://arxiv.org/abs/1804.08460)). 
It contains the code to replicate the experiments and train the models described in the paper.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
 

Please use the following citation:

```
@inproceedings{TUD-CS-2018-01,
    title = {Mixing Context Granularities for Improved Entity Linking on Question Answering Data across Entity Categories},
    author = {Sorokin, Daniil and Gurevych, Iryna},
    publisher = {Association for Computational Linguistics},
    booktitle = {Proceedings of the 7th Joint Conference on Lexical and Computational Semantics (*SEM 2018) },
    pages = {to appear},
    month = jun,
    year = {2018},
    location = {New Orleans, LA, U.S.}
}
```

### Paper abstract:
> The first stage of every knowledge base question answering approach is to link entities in the input question. 
  We investigate entity linking in the context of a question answering task and present a jointly optimized neural architecture for entity mention detection and entity disambiguation that models the surrounding context on different levels of granularity. 

> We use the Wikidata knowledge base and available question answering datasets to create benchmarks for entity linking on question answering data. 
  Our approach outperforms the previous state-of-the-art system on this data, resulting in an average 8% improvement of the final score. We further demonstrate that our model delivers a strong performance across different entity categories.

Please, refer to the paper for more the model description and training details 
 
### Contacts:
If you have any questions regarding the code, please, don't hesitate to contact the authors or report an issue.
  * Daniil Sorokin, \<lastname\>@ukp.informatik.tu-darmstadt.de
  * https://www.ukp.tu-darmstadt.de
  * https://www.tu-darmstadt.de

### Project structure:

<table>
    <tr>
        <th>File</th><th>Description</th>
    </tr>
    <tr>
        <td>configs/</td><td>Configuration files for the experiments</td>
    </tr>
    <tr>
        <td>entitylinking/core</td><td>Mention extraction and candidate retrieval</td>
    </tr>
    <tr>
        <td>entitylinking/datasets</td><td>Datasets IO</td>
    </tr>
    <tr>
        <td>entitylinking/evaluation</td><td>Evaluation measures and scripts</td>
    </tr>
    <tr>
        <td>entitylinking/mlearning</td><td>Model definition and training scripts</td>
    </tr>
    <tr>
        <td>entitylinking/wikidata</td><td>Retrieving information from Wikidata</td>
    </tr>
    <tr>
        <td>resources/</td><td>Necessary resources</td>
    </tr>
    <tr>
        <td>trainedmodels/</td><td>Trained models</td>
    </tr>
</table>


#### Requirements:
* Python 3.6
* PyTorch 0.3.0 - [read here about installation](http://pytorch.org/)
* See `requirements.txt` for the full list of packages

### Installation:

1. Download and install Anaconda (https://www.anaconda.com/)
2. Create an anaconda environment: `conda create -n qa-env python=3.6` and activate it `conda activate qa-env`
3. Install PyTorch 0.3.1: `conda install pytorch=0.3.1 -c pytorch` (with CUDA if you want to use GPU)
4. Install the rest of the dependencies from the `requirements.txt` with: `conda install --yes --file requirements.txt`. 
5. Install `pycorenlp, SPARQLWrapper` with `pip install pycorenlp SPARQLWrapper`.
6. Create a local copy of the Wikidata knowledge base in RDF format. We use the [Virtuoso Opensource Server](https://github.com/openlink/virtuoso-opensource). See [here](WikidataHowTo.md) for more info on the Wikidata installation. (This step takes a lot of time!). Right now this is the only way to run the models at test time, we are working to providing a smaller Wikidata dump just for the training/evaluation on the data sets.

### Running the experiments from the paper:

See `run_experiments.sh`

### Using the pre-trained model:

Follow the steps to use this project as an external entity-linking tool. `FeatureModel_Baseline` is a part of the repository, you can download the `VCG` model [here](https://public.ukp.informatik.tu-darmstadt.de/starsem18-entity-linking/VectorModel_VCG.zip).

For the VCG model you also need KB embeddings produced by [Fast-TransX](https://github.com/thunlp/Fast-TransX). Download [here](https://public.ukp.informatik.tu-darmstadt.de/starsem18-entity-linking/Wikidata_TransE_50.zip). 

1. Clone/Download the project
2. Take a pre-trained model and extract it into a `trainedmodels/` folder in the main directory of the project
3. Download the [GloVe embeddings, glove.6B.zip](https://nlp.stanford.edu/projects/glove/)
and put them into the folder `resources/glove/` in the main directory of the project
4. Modify the path to the word embeddings in the configuration file for the model: `trainedmodels/FeatureModel_Baseline.param`
5. Make sure that the project folder in your Python PATH
6. Use the following code to initialize an entity linker and apply it on new data:

```python
from entitylinking import core
    
entitylinker = core.MLLinker(path_to_model="trainedmodels/FeatureModel_Baseline.torchweights")
output = entitylinker.link_entities_in_raw_input("Barack Obama is a president.")
print(output.entities)
```



### License:
* Apache License Version 2.0
