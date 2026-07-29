from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import URLError

import cv2 as cv
import numpy as np
import pytest

from dip_toolkit.modules.image_loader import ImageLoader


def create_test_image(path: Path, value: int = 180) -> np.ndarray:
    """Cria uma imagem pequena cujo canal verde identifica o arquivo."""
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[:, :, 1] = value
    assert cv.imwrite(str(path), image)
    return image


def encoded_test_image() -> bytes:
    """Retorna uma imagem PNG em memória para testes de URL."""
    image = np.full((8, 12, 3), 90, dtype=np.uint8)
    ok, encoded = cv.imencode(".png", image)
    assert ok
    return encoded.tobytes()


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


def test_load_image_rejects_undecodable_file(tmp_path: Path) -> None:
    invalid_image = tmp_path / "invalid.png"
    invalid_image.write_text("not an image")

    with pytest.raises(ValueError, match="Não foi possível decodificar"):
        ImageLoader().load_image(invalid_image)


def test_load_images_keeps_input_order(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    create_test_image(first, value=10)
    create_test_image(second, value=20)

    images = ImageLoader().load_images([second, first])

    assert [image[0, 0, 1] for image in images] == [20, 10]


def test_load_images_can_skip_invalid_files(tmp_path: Path) -> None:
    valid_image = tmp_path / "valid.png"
    create_test_image(valid_image)

    images = ImageLoader().load_images(
        [tmp_path / "missing.png", valid_image],
        skip_invalid=True,
    )

    assert len(images) == 1


def test_load_images_rejects_a_single_path(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    create_test_image(image_path)

    with pytest.raises(TypeError, match="sequência de caminhos"):
        ImageLoader().load_images(image_path)


def test_load_directory_uses_alphabetical_order(tmp_path: Path) -> None:
    create_test_image(tmp_path / "b.png", value=20)
    create_test_image(tmp_path / "a.png", value=10)
    (tmp_path / "notes.txt").write_text("not an image")

    images = ImageLoader().load_images_from_directory(tmp_path)

    assert [image[0, 0, 1] for image in images] == [10, 20]


def test_load_directory_can_search_recursively(tmp_path: Path) -> None:
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    create_test_image(tmp_path / "root.png", value=10)
    create_test_image(nested_directory / "nested.png", value=20)

    images = ImageLoader().load_images_from_directory(tmp_path, recursive=True)

    assert [image[0, 0, 1] for image in images] == [20, 10]


def test_load_directory_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Diretório não encontrado"):
        ImageLoader().load_images_from_directory(tmp_path / "missing")


def test_load_image_from_url_reads_encoded_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = MagicMock()
    response.read.return_value = encoded_test_image()
    context_manager = MagicMock()
    context_manager.__enter__.return_value = response
    mock_urlopen = MagicMock(return_value=context_manager)
    monkeypatch.setattr("dip_toolkit.modules.image_loader.urlopen", mock_urlopen)

    image = ImageLoader().load_image_from_url(
        "https://example.com/image.png",
        timeout_seconds=12.5,
    )

    assert image.shape == (8, 12, 3)
    assert mock_urlopen.call_args.kwargs["timeout"] == 12.5


def test_load_image_from_url_rejects_invalid_url() -> None:
    with pytest.raises(ValueError, match="HTTP"):
        ImageLoader().load_image_from_url("image.png")


def test_load_image_from_url_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="maior que zero"):
        ImageLoader().load_image_from_url(
            "https://example.com/image.png",
            timeout_seconds=0,
        )


def test_load_image_from_url_wraps_network_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_url_error(*args: object, **kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr("dip_toolkit.modules.image_loader.urlopen", raise_url_error)

    with pytest.raises(RuntimeError, match="Não foi possível baixar"):
        ImageLoader().load_image_from_url("https://example.com/image.png")


def test_load_image_from_url_rejects_invalid_image_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = MagicMock()
    response.read.return_value = b"not an image"
    context_manager = MagicMock()
    context_manager.__enter__.return_value = response
    monkeypatch.setattr(
        "dip_toolkit.modules.image_loader.urlopen",
        MagicMock(return_value=context_manager),
    )

    with pytest.raises(ValueError, match="não pôde ser decodificado"):
        ImageLoader().load_image_from_url("https://example.com/image.png")
