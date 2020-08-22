from wikidata_linker_utils.type_collection import TypeCollection
import json
import numpy as np
from os import makedirs
from os.path import join, realpath, dirname, exists, expanduser
from absl import logging

# def convert_types_to_disambiguator_config(
#     all_types,
#     output_name,
#     output_dirname,
#     config_path,
#     wiki=None,
#     lang=None,
#     min_count=2,
#     sample_size=9999999,
#     min_percent=0.001,
# ):
#     output_dir = join(output_name, output_dirname)
#     config = {
#         "wiki": wiki,
#         "min_count": min_count,
#         "min_percent": min_percent,
#         "wikidata": "wikidata",
#         "prefix": lang + "wiki",
#         "sample_size": sample_size,
#         "num_names_to_load": 4000,
#         "language_path": lang + "_trie_fixed",
#         "redirections": lang + "_redirections.tsv",
#         "classification": [],
#     }
#     for types in all_types:
#         typename = types["qid"]
#         relation = types["relation"]
#         surname = typename + "_" + relation.replace(" ", "_")
#         type_dir = join(output_dir, surname + "_classification")
#         config["classification"].append(type_dir)

#     with open(config_path, "wt") as fout:
#         json.dump(config, fout)


def convert_types_to_model_config(
    type_collection,
    all_types, 
    output_dir,
    languages = ['en']
):

    config = {
        # "wikidata_path": wikidata_path,
        # "classification_path": output_dirname,
        "features": [
            {"type": "word", "dimension": 100, "max_vocab": 1_000_000},
            {"type": "suffix", "length": 2, "dimension": 6, "max_vocab": 1_000_000},
            {"type": "suffix", "length": 3, "dimension": 6, "max_vocab": 1_000_000},
            {"type": "prefix", "length": 2, "dimension": 6, "max_vocab": 1_000_000},
            {"type": "prefix", "length": 3, "dimension": 6},
            {"type": "digit"},
            {"type": "uppercase"},
            {"type": "punctuation_count"},
        ],
        "datasets": [],
        "objectives": [],
    }

    for types in all_types:
        typename = types["qid"]
        relation = types["relation"]
        surname = typename + "_" + relation.replace(" ", "_")


        # create sub type classifications
        type_dir = join(output_dir, surname + "_classification")
        
        logging.info('generating %s', type_dir)
        makedirs(type_dir, exist_ok=True)

        if not exists(join(type_dir, "classification.npy")):
            np.save(
                join(type_dir, "classification.npy"),
                type_collection.satisfy([relation], [type_collection.name2index[typename]]),
            )

        with open(join(type_dir, "classes.txt"), "wt") as fout:
            fout.write("N\nY\n")

        config["objectives"].append(
            {
                "name": surname,
                "type": "softmax",
                "vocab": join(
                    output_dir, surname + "_classification", "classes.txt"
                ),
            }
        )

    # create dev and train for each language
    datasets = [("{}_train.h5".format(lang), "train") for lang in languages] + [
        ("{}_dev.h5".format(lang), "dev") for lang in languages
    ]
    for dataset, dtype in datasets:
        config["datasets"].append(
            {
                "type": dtype,
                "path": dataset,
                "x": 0,
                "ignore": "other",
                "y": [
                    {
                        "column": 1,
                        "objective": obj["name"],
                        "classification": obj["name"] + "_classification",
                    }
                    for obj in config["objectives"]
                ],
            }
        )
    return config


def main():
    """
    The output of evolve_type_system.py is a set of types (root + relation) 
    that can be used to build a type system. To create a config file that can 
    be used to train an LSTM use the jupyter notebook 
    extraction/TypeSystemToNeuralTypeSystem.ipynb.
    """

    output_dirname = 'test_lang'
    output_name = '/tmp/test/en_small_train_config.json'

    # load wikidata types
    wikidata_location = expanduser('~/content/models/deeptype/wikidata')
    blacklist_location = expanduser('~/content/src/ranking-research/docker/deeptype/deeptype/extraction/blacklist.json')
    logging.info('loading type system')
    type_collection = TypeCollection(wikidata_location, num_names_to_load=0)
    type_collection.load_blacklist(blacklist_location)

    # export your GA model by passing the path to the report file from evolve_type_system,
    # and path to wikidata & training data path (exported wikipedia text in h5 format) along
    # with the desired name for the model
    # search2model(
    #     collection=c,
    #     dataset_path="training_data",
    #     report_path="ga-0.00007.json",
    #     wikidata_path="data/wikidata",
    #     name="ga_model",
    #     languages=['en']
    # )
    
    type_path = '/home/fabriziomilo/content/src/ranking-research/docker/deeptype/deeptype/report.json'
    with open(type_path, "rt") as fin:
        all_types = json.load(fin)

    output_dir = join(dirname(output_name), output_dirname)
    makedirs(output_dir, exist_ok=True)
    
    languages = ['en']

    config = convert_types_to_model_config(
        type_collection,
        all_types,
        output_dir,
        languages,
    )

    config["wikidata_path"] = wikidata_path
    config["classification_path"] = output_dirname

    with open(output_name, "wt") as fout:
        json.dump(config, fout, indent=4)

    name = 'en'
    for lang in languages:
        # convert_types_to_disambiguator_config(
        #     all_types,
        #     dataset_path,
        #     directory,
        #     "extraction/{name}_disambiguator_config_export.json".format(
        #         name=name, lang=lang
        #     ),
        min_percent = 0
        min_count=0
        sample_size=1000
        wiki="{}wiki-latest-pages-articles.xml".format(lang),
        
        output_dir = join(output_name, output_dirname)
        
        config = {
            "wiki": wiki,
            "min_count": min_count,
            "min_percent": min_percent,
            "wikidata": "wikidata",
            "prefix": lang + "wiki",
            "sample_size": sample_size,
            "num_names_to_load": 4000,
            "language_path": lang + "_trie_fixed",
            "redirections": lang + "_redirections.tsv",
            "classification": [],
        }
        for types in all_types:
            typename = types["qid"]
            relation = types["relation"]
            surname = typename + "_" + relation.replace(" ", "_")
            type_dir = join(output_dir, surname + "_classification")
            config["classification"].append(type_dir)

        with open(config_path, "wt") as fout:
            json.dump(config, fout)


if __name__ == "__main__":
    main()
