from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from dip_toolkit.modules.image_loader import ImageLoader


def create_test_image(path: Path) -> np.ndarray:
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    assert cv.imwrite(str(path), image)
    return image


def test_load_image_reads_color_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    expected = create_test_image(image_path)

    loaded = ImageLoader().load_image(image_path)

    assert loaded.shape == expected.shape
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, expected)


def test_load_image_reads_grayscale(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    create_test_image(image_path)

    loaded = ImageLoader().load_image(image_path, flags=cv.IMREAD_GRAYSCALE)

    assert loaded.ndim == 2


def test_load_image_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
        ImageLoader().load_image(tmp_path / "missing.png")


def test_load_directory_uses_alphabetical_order(tmp_path: Path) -> None:
    create_test_image(tmp_path / "b.png")
    create_test_image(tmp_path / "a.png")

    images = ImageLoader().load_images_from_directory(tmp_path)

    assert len(images) == 2
