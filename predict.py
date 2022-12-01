# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import io
import os
import shutil
import typing

from PIL import Image, ImageOps
from cog import BasePredictor, Path, File


class Predictor(BasePredictor):
    model_512 = {}
    model_768 = {}
    inpainting_model = {}
    upscaling_model = {}

    def _setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("running setup...")

        from scripts import txt2img
        from scripts.gradio import inpainting, superresolution

        self.model_512["model"] = txt2img.load_models(
            txt2img.parse_args([
                "--ckpt", "checkpoints/512-base-ema.ckpt",
                "--config", "configs/stable-diffusion/v2-inference.yaml"
            ])
        )

        self.model_768["model"] = txt2img.load_models(
            txt2img.parse_args([
                "--ckpt", "checkpoints/768-v-ema.ckpt",
                "--config", "configs/stable-diffusion/v2-inference-v.yaml"
            ])
        )

        inpainting.sampler = inpainting.initialize_model(
            config="configs/stable-diffusion/v2-inpainting-inference.yaml",
            ckpt="checkpoints/512-inpainting-ema.ckpt"
        )
        self.inpainting_model = vars(inpainting.sampler)

        superresolution.sampler = superresolution.initialize_model(
            config="configs/stable-diffusion/x4-upscaling.yaml",
            ckpt="checkpoints/x4-upscaler-ema.ckpt"
        )
        self.upscaling_model = vars(superresolution.sampler)

        move_models(self.model_512, "cpu")
        move_models(self.model_768, "cpu")
        move_models(self.inpainting_model, "cpu")
        move_models(self.upscaling_model, "cpu")


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
        negative_prompt: str = "",
    ) -> typing.List[File]:
        """Run a single prediction on the model"""

        if not self.model_512:
            self._setup()

        from scripts import txt2img, img2img
        from scripts.gradio import inpainting, superresolution

        if not seed:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if edit_image:
            with models_in_gpu(self.inpainting_model):
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
                    negative_prompt=negative_prompt,
                )
        else:
            args = [
                "--ckpt", "None",
                "--config", "None",
                "--prompt", prompt,
                "--n_iter", "1",
                "--n_samples", str(num_outputs),
                "--scale", str(guidance_scale),
                "--seed", str(seed),
                "--negative_prompt", negative_prompt,
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

            if width > 512 or height > 512:
                model_dict = self.model_768
            else:
                model_dict = self.model_512

            print("running with args:", " ".join(map(str, args)))
            with models_in_gpu(model_dict):
                results = module.run_models(module.parse_args(args), **model_dict)

        if upscaling_inference_steps:
            with models_in_gpu(self.upscaling_model):
                results = [
                    superresolution.predict(
                        input_image=ImageOps.contain(input_image, (512, 512)),
                        prompt=prompt,
                        steps=upscaling_inference_steps,
                        num_samples=1,
                        scale=guidance_scale,
                        seed=seed,
                        eta=0,
                        noise_level=20,
                        negative_prompt=negative_prompt,
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

