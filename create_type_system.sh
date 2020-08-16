#!/usr/bin/env bash
export LANGUAGE=en
export DATA_DIR=$HOME/content/models/deeptype/
python3 extraction/project_graph.py ${DATA_DIR}wikidata/ extraction/classifiers/type_classifier.py