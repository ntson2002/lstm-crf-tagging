#!/usr/bin/env bash

cd $HOME/Bitbucket/lstm-crf-tagging/lstm-tagger-v4/
export PYTHONPATH=$PWD:$PYTHONPATH
python ./apps/jpl-rre/rre-api.py