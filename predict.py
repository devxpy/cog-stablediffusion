# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import shutil

import typing
from glob import glob

from cog import BasePredictor, Input, Path
from scripts import txt2img, img2img


class Predictor(BasePredictor):
    models_dict = {}

    def _setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.models_dict = txt2img.load_models(
            txt2img.parse_args(
                ["--ckpt", "checkpoints/768-v-ema.ckpt", "--config", "configs/stable-diffusion/v2-inference-v.yaml"]
            )
        )
        move_models(self.models_dict, "cpu")

    def predict(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_outputs: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: int = None,
        init_image: Path = None,
    ) -> typing.List[Path]:
        """Run a single prediction on the model"""

        if not self.models_dict:
            self._setup()

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        args = [
            "--ckpt", "checkpoints/768-v-ema.ckpt",
            "--config", "configs/stable-diffusion/v2-inference-v.yaml",
            "--prompt", prompt,
            "--W", str(width),
            "--H", str(height),
            "--steps", str(num_inference_steps),
            "--n_iter", "1",
            "--n_samples", str(num_outputs),
            "--scale", str(guidance_scale),
            "--seed", str(seed),
            "--outdir", "outputs/",
        ]
        if init_image:
            args += [
                "--init-img", str(init_image),
                "--strength", str(strength),
            ]
            module = img2img
        else:
            module = txt2img

        shutil.rmtree('outputs/', ignore_errors=True)

        with models_in_gpu(self.models_dict):
            module.run_models(module.parse_args(args), **self.models_dict)

        return [Path(path) for path in glob("outputs/samples/*.png")]


import torch
import contextlib


@contextlib.contextmanager
def models_in_gpu(models_dict):
    move_models(models_dict, "cuda")
    try:
        yield
    finally:
        move_models(models_dict, "cpu")


def move_models(models_dict, device):
    for attr, value in models_dict.items():
        try:
            value = value.to(device)
        except AttributeError:
            pass
        else:
            print(f"Moving {attr} to {device}")
            models_dict[attr] = value
    torch.cuda.empty_cache()

