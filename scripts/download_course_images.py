"""Baixa as imagens públicas da disciplina para o diretório local assets/."""

from __future__ import annotations

import argparse
from pathlib import Path

from dip_toolkit.assets import download_course_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/images"),
        help="Diretório de destino (padrão: assets/images).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Solicita um novo download mesmo se já houver arquivos.",
    )
    args = parser.parse_args()

    destination = download_course_image(
        "cameraman_original.png", output_dir=args.output, force=args.force
    )
    print(f"Imagem disponível em: {destination}")


if __name__ == "__main__":
    main()
