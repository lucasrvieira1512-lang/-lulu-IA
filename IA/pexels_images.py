"""Busca e download de fotos usando a API oficial do Pexels."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
MAX_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class PexelsPhoto:
    path: Path
    alt: str
    photographer: str
    page_url: str


def search_pexels_image(query: str, output_directory: Path) -> PexelsPhoto:
    """Search Pexels for one matching photo and save its large preview."""
    query = query.strip()
    if not query:
        raise ValueError("Descreva a foto que quer buscar.")
    if len(query) > 200:
        raise ValueError("A busca de fotos deve ter até 200 caracteres.")

    api_key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Para buscar fotos reais, crie uma chave gratuita em "
            "https://www.pexels.com/api/ e configure a variável PEXELS_API_KEY "
            "no Windows. Depois, reinicie o assistente."
        )

    request = Request(
        f"{PEXELS_SEARCH_URL}?{urlencode({'query': query, 'per_page': 1, 'locale': 'pt-BR'})}",
        headers={"Authorization": api_key, "User-Agent": "AssistenteLocal/1.0"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code in (401, 403):
            raise RuntimeError(
                "O Pexels recusou a chave de API. Confira a variável PEXELS_API_KEY."
            ) from error
        if error.code == 429:
            raise RuntimeError(
                "O limite de buscas do Pexels foi atingido. Tente novamente mais tarde."
            ) from error
        raise RuntimeError(f"A API do Pexels respondeu com erro HTTP {error.code}.") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError("Não consegui conectar à API do Pexels. Verifique sua internet.") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("A resposta da API do Pexels veio em um formato inválido.") from error

    photos = payload.get("photos") if isinstance(payload, dict) else None
    if not isinstance(photos, list) or not photos:
        raise RuntimeError(f"Não encontrei fotos no Pexels para “{query}”.")

    photo = photos[0]
    if not isinstance(photo, dict):
        raise RuntimeError("A API do Pexels retornou uma foto em formato inválido.")
    src = photo.get("src")
    image_url = src.get("large") if isinstance(src, dict) else None
    photographer = photo.get("photographer")
    page_url = photo.get("url")
    alt = photo.get("alt") or query
    if not all(isinstance(value, str) and value.strip() for value in (image_url, photographer, page_url)):
        raise RuntimeError("A API do Pexels retornou dados incompletos para a foto.")
    if not image_url.startswith("https://"):
        raise RuntimeError("A API do Pexels retornou um endereço de imagem inválido.")

    image_request = Request(image_url, headers={"User-Agent": "AssistenteLocal/1.0"})
    try:
        with urlopen(image_request, timeout=30) as response:
            content_type = response.headers.get_content_type()
            if not content_type.startswith("image/"):
                raise RuntimeError("O endereço retornado pelo Pexels não contém uma imagem.")
            image_data = response.read(MAX_IMAGE_BYTES + 1)
    except HTTPError as error:
        raise RuntimeError(f"Não consegui baixar a foto do Pexels (HTTP {error.code}).") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError("Não consegui baixar a foto do Pexels. Verifique sua internet.") from error

    if len(image_data) > MAX_IMAGE_BYTES:
        raise RuntimeError("A foto retornada pelo Pexels excede o limite de 20 MB.")
    if not image_data:
        raise RuntimeError("O Pexels retornou uma foto vazia.")

    output_directory.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("pexels_%Y%m%d_%H%M%S_%f.jpg")
    output_path = output_directory / filename
    output_path.write_bytes(image_data)
    return PexelsPhoto(
        path=output_path,
        alt=alt.strip() if isinstance(alt, str) else query,
        photographer=photographer.strip(),
        page_url=page_url.strip(),
    )
