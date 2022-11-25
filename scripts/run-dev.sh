#!/usr/bin/env bash

NAME=stablediffusion2-dev

set -x

docker build . -t $NAME
docker run \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -p 6011:5000 \
  --gpus all \
  $NAME
