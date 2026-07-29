"""Carregamento de imagens locais, remotas e em lote com OpenCV."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2 as cv
import numpy as np


class ImageLoader:
    """Carrega imagens de arquivos locais, diretórios ou URLs diretas."""

    _VALID_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    def load_image(
        self,
        filename: str | Path,
        flags: int = cv.IMREAD_COLOR,
    ) -> np.ndarray:
        """Carrega uma imagem local e valida se ela pôde ser decodificada."""
        path = Path(filename).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        image = cv.imread(str(path), flags)
        if image is None:
            raise ValueError(f"Não foi possível decodificar a imagem: {path}")

        return image

    def load_images(
        self,
        filenames: Iterable[str | Path],
        flags: int = cv.IMREAD_COLOR,
        *,
        skip_invalid: bool = False,
    ) -> list[np.ndarray]:
        """Carrega uma sequência de imagens.

        Por padrão, a primeira falha interrompe a execução. Use
        ``skip_invalid=True`` quando o comportamento desejado for ignorar
        arquivos inválidos.
        """
        images: list[np.ndarray] = []
        for filename in filenames:
            try:
                images.append(self.load_image(filename, flags=flags))
            except (FileNotFoundError, ValueError):
                if not skip_invalid:
                    raise
        return images

    def load_images_from_directory(
        self,
        directory: str | Path,
        flags: int = cv.IMREAD_COLOR,
        *,
        recursive: bool = False,
        skip_invalid: bool = False,
    ) -> list[np.ndarray]:
        """Carrega imagens de um diretório em ordem alfabética."""
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Diretório não encontrado: {root}")

        iterator = root.rglob("*") if recursive else root.iterdir()
        files = sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in self._VALID_EXTENSIONS
        )

        return self.load_images(
            files,
            flags=flags,
            skip_invalid=skip_invalid,
        )

    def load_image_from_url(
        self,
        url: str,
        flags: int = cv.IMREAD_COLOR,
        *,
        timeout_seconds: float = 30.0,
    ) -> np.ndarray:
        """Carrega uma imagem a partir de uma URL direta para o arquivo."""
        if not url.startswith(("http://", "https://")):
            raise ValueError("A URL deve começar com http:// ou https://")

        request = Request(url, headers={"User-Agent": "dip-toolkit/0.1"})

        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Não foi possível baixar a imagem: {url}") from exc

        image_data = np.frombuffer(payload, dtype=np.uint8)
        image = cv.imdecode(image_data, flags)
        if image is None:
            raise ValueError(
                "O conteúdo baixado não pôde ser decodificado como imagem. "
                "Confirme se a URL aponta diretamente para um arquivo."
            )

        return image
