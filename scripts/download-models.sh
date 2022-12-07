#!/usr/bin/env bash

set -ex

mkdir -p checkpoints
cd checkpoints

wget -c 'https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt'
wget -c 'https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt'
wget -c 'https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt'
wget -c 'https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/x4-upscaler-ema.ckpt'

wget -c 'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt'
wget -c 'https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt'