#!/usr/bin/env bash

NAME=stablediffusion2

set -x

docker rm -f $NAME

docker run -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -p 5020:5000 \
  --gpus all \
  stablediffusion2:v2

docker logs -f $NAME
