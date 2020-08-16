#!/usr/bin/env bash
set -x
set -e 

export LANGUAGE=en
export DATA_DIR=$HOME/content/models/deeptype/
export CLASSIFICATION_DIR=$HOME/content/datasets/deeptype-classification/
mkdir -p ${CLASSIFICATION_DIR}

python3 extraction/produce_wikidata_tsv.py \
    extraction/configs/en_disambiguator_config_export_small.json \
    --relative_to ${DATA_DIR} sample_data.tsv

python3 learning/evaluate_learnability.py \
    --dataset sample_data.tsv \
    --out report.json \
    --wikidata ${DATA_DIR}wikidata/
