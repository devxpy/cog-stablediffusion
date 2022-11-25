#!/usr/bin/env bash

set -ex

mkdir checkpoints
cd checkpoints

wget 'https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt'
wget 'https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt'
wget 'https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/x4-upscaler-ema.ckpt'
