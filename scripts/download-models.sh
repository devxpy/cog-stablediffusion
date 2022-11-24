#!/usr/bin/env bash

set -ex

mkdir checkpoints
cd checkpoints
wget 'https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt'
