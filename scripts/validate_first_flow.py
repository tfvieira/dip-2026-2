"""Valida localmente o fluxo Drive → ImageLoader → saída processada."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv

from dip_toolkit import download_course_image
from dip_toolkit.modules.image_loader import ImageLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("assets/images"),
    )
    parser.add_argument(
        "--filename",
        default="cameraman_original.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cameraman_preview.png"),
    )
    args = parser.parse_args()

    image_path = download_course_image(args.filename, output_dir=args.images_dir)
    image = ImageLoader().load_image(image_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(args.output), image):
        raise RuntimeError(f"Não foi possível salvar a saída em: {args.output}")

    print("Fluxo validado com sucesso.")
    print(f"Entrada: {image_path}")
    print(f"Formato: {image.shape}")
    print(f"Tipo: {image.dtype}")
    print(f"Prévia salva em: {args.output.resolve()}")


if __name__ == "__main__":
    main()
