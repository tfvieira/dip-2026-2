"""Utilitários para acesso às imagens públicas da disciplina."""

from pathlib import Path

import gdown
import requests

IMAGE_RESOLVER_URL = "https://script.google.com/macros/s/AKfycbwT4MLm2Deh2f_FvlI0ie0JipZSuxZXfs_DDfp3s_mYQQgOK5mlMbl1JMif-9RR9rrx/exec"

DEFAULT_IMAGES_DIR = Path("assets/images")


def download_course_image(
    filename: str,
    output_dir: str | Path = DEFAULT_IMAGES_DIR,
    *,
    force: bool = False,
) -> Path:
    """Baixa uma imagem da disciplina pelo nome."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    destination = output_dir / filename

    if destination.exists() and not force:
        return destination

    response = requests.get(
        IMAGE_RESOLVER_URL,
        params={"name": filename},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if not data.get("ok"):
        raise FileNotFoundError(data.get("error"))

    result = gdown.download(
        id=data["id"],
        output=str(destination),
        quiet=False,
    )

    if not result or not destination.exists():
        raise RuntimeError(f"Não foi possível baixar '{filename}'.")

    return destination
