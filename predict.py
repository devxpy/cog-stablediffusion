# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import io
import os
import shutil
import typing

from PIL import Image, ImageOps
from cog import BasePredictor, Path, File


class Predictor(BasePredictor):
    models = {}
    inpainting_models = {}
    upscaling_models = {}

    def _setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("running setup...")

        from scripts import txt2img, img2img
        from scripts.gradio import inpainting, superresolution

        self.models["model"] = txt2img.load_models(
            txt2img.parse_args([
                "--ckpt", "checkpoints/768-v-ema.ckpt",
                "--config", "configs/stable-diffusion/v2-inference-v.yaml"
            ])
        )

        inpainting.sampler = inpainting.initialize_model(
            config="configs/stable-diffusion/v2-inpainting-inference.yaml",
            ckpt="checkpoints/512-inpainting-ema.ckpt"
        )
        self.inpainting_models["model"] = inpainting.sampler.model

        superresolution.sampler = superresolution.initialize_model(
            config="configs/stable-diffusion/x4-upscaling.yaml",
            ckpt="checkpoints/x4-upscaler-ema.ckpt"
        )
        self.upscaling_models["model"] = superresolution.sampler.model

        move_models(self.models, "cpu")
        move_models(self.inpainting_models, "cpu")
        move_models(self.upscaling_models, "cpu")


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
        edit_image: Path = None,
        mask_image: Path = None,
        sampler: str = "ddim",
        upscaling_inference_steps: int = 0,
    ) -> typing.List[File]:
        """Run a single prediction on the model"""

        from scripts import txt2img, img2img
        from scripts.gradio import inpainting, superresolution

        if not self.models:
            self._setup()

        if not seed:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if edit_image:
            with models_in_gpu(self.inpainting_models):
                results = inpainting.predict(
                    input_image={
                        "image": Image.open(edit_image),
                        "mask": ImageOps.invert(Image.open(mask_image)),
                    },
                    prompt=prompt,
                    ddim_steps=num_inference_steps,
                    num_samples=num_outputs,
                    scale=guidance_scale,
                    seed=seed,
                )
        else:
            args = [
                "--ckpt", "checkpoints/768-v-ema.ckpt",
                "--config", "configs/stable-diffusion/v2-inference-v.yaml",
                "--prompt", prompt,
                "--n_iter", "1",
                "--n_samples", str(num_outputs),
                "--scale", str(guidance_scale),
                "--seed", str(seed),
            ]

            if sampler == "plms":
                args += ["--plms"]
            elif sampler == "dpm":
                args += ["--dpm"]

            if init_image:
                args += [
                    "--init-img", str(init_image),
                    "--strength", str(strength),
                    "--ddim_steps", str(num_inference_steps),
                ]
                module = img2img
            else:
                args += [
                    "--W", str(width),
                    "--H", str(height),
                    "--steps", str(num_inference_steps),
                ]
                module = txt2img

            shutil.rmtree('outputs/', ignore_errors=True)

            print("running with args:", " ".join(map(str, args)))
            with models_in_gpu(self.models):
                results = module.run_models(module.parse_args(args), **self.models)

        if upscaling_inference_steps:
            with models_in_gpu(self.upscaling_models):
                results = [
                    superresolution.predict(
                        input_image=input_image,
                        prompt=prompt,
                        steps=upscaling_inference_steps,
                        num_samples=num_outputs,
                        scale=guidance_scale,
                        seed=seed,
                        eta=0,
                        noise_level=20,
                    )[0]
                    for input_image in results
                ]

        ret = []
        for img in results:
            f = io.BytesIO()
            img.save(f, format='PNG')
            ret.append(f)
        return ret


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

