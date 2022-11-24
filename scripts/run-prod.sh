#!/usr/bin/env bash

NAME=stablediffusion2

set -x

docker rm -f $NAME

docker run -d --restart always \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -p 5011:5000 \
  --gpus all \
  $NAME

docker logs -f $NAME
