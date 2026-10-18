"""Geração local de conceitos visuais para jardins."""

from __future__ import annotations

import re
import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_pipeline_state: dict[str, Any] = {"instance": None}
_pipeline_lock = threading.Lock()


def extract_image_prompt(message: str) -> str | None:
    normalized = unicodedata.normalize("NFKD", message)
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower().strip()
    match = re.match(
        r"^(?:(?:quero que (?:voce )?)?(?:voce pode\s+)?"
        r"(?:gere|gera|crie|cria|desenhe|desenha|faca|faz|fazer|criar)\s+"
        r"(?:uma?\s+)?(?:imagem|foto))"
        r"(?:\s+(?:de|mostrando|com))?\s*[:\-]?\s*(.+)$",
        normalized,
    )
    return match.group(1).strip() if match else None


def _build_photorealistic_prompt(prompt: str) -> str:
    return (
        f"{prompt}. Wide-angle outdoor landscape photo; ultra-photorealistic, natural "
        "daylight, lifelike plants and textures, true-to-life colors, detailed composition. "
        "Preserve the requested subject and colors; real photograph, not CGI or illustration."
    )


def generate_image(prompt: str, output_directory: Path) -> Path:
    """Generate a photorealistic image with the local SDXL Base model."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Descreva a imagem de jardim que quer criar.")
    if len(prompt) > 1000:
        raise ValueError("A descrição da imagem deve ter até 1.000 caracteres.")

    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
    except ImportError as error:
        raise RuntimeError(
            "O gerador local não está instalado neste Python. Instale torch, diffusers, "
            "transformers, accelerate, safetensors e pillow."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError("O gerador precisa de PyTorch com CUDA e uma GPU NVIDIA disponível.")

    with _pipeline_lock:
        pipeline = _pipeline_state["instance"]
        if pipeline is None:
            try:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    MODEL_ID,
                    dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                )
                pipeline.enable_model_cpu_offload()
                pipeline.vae.enable_slicing()
                _pipeline_state["instance"] = pipeline
            except Exception as error:
                _pipeline_state["instance"] = None
                raise RuntimeError(
                    "Não consegui carregar o SDXL Base. Verifique a conexão e tente novamente."
                ) from error

        try:
            negative_prompt = (
                "illustration, painting, drawing, anime, cartoon, 3d, cgi, render, "
                "plastic texture, artificial-looking materials, oversaturated colors, "
                "blurry, out of focus, low resolution, jpeg artifacts, text, watermark, "
                "logo, frame"
            )
            if "jardim" in prompt.lower() or "garden" in prompt.lower():
                negative_prompt += ", person, people, portrait, indoor, living room"

            result = pipeline(
                prompt=_build_photorealistic_prompt(prompt),
                negative_prompt=negative_prompt,
                width=1024,
                height=768,
                num_inference_steps=35,
                guidance_scale=7.0,
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "A GPU ficou sem memória para gerar a imagem. Feche outros programas que usam a GPU e tente novamente."
            ) from error

    output_directory.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("jardim_%Y%m%d_%H%M%S_%f.png")
    output_path = output_directory / filename
    result.images[0].save(output_path)
    return output_path