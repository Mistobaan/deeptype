#!/usr/bin/env bash
set -e 
set -x

export LANGUAGE=en
export DATA_DIR=$HOME/content/models/deeptype/
export CLASSIFICATION_DIR=$HOME/content/datasets/deeptype-classification/
mkdir -p ${CLASSIFICATION_DIR}
python3 extraction/project_graph.py \
    ${DATA_DIR}wikidata/ \
    extraction/classifiers/type_classifier.py \
    --export_classification ${CLASSIFICATION_DIR}