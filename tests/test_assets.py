from pathlib import Path
from unittest.mock import Mock

import pytest

from dip_toolkit.assets import download_course_image


def test_returns_existing_image_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "cameraman_original.png"
    destination.write_bytes(b"existing-image")

    request_get = Mock()
    gdown_download = Mock()

    monkeypatch.setattr(
        "dip_toolkit.assets.requests.get",
        request_get,
    )
    monkeypatch.setattr(
        "dip_toolkit.assets.gdown.download",
        gdown_download,
    )

    result = download_course_image(
        "cameraman_original.png",
        output_dir=tmp_path,
    )

    assert result == destination
    request_get.assert_not_called()
    gdown_download.assert_not_called()


def test_downloads_image_by_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.json.return_value = {
        "ok": True,
        "id": "google-drive-file-id",
    }

    monkeypatch.setattr(
        "dip_toolkit.assets.requests.get",
        Mock(return_value=response),
    )

    def fake_download(
        *,
        id: str,
        output: str,
        quiet: bool,
    ) -> str:
        assert id == "google-drive-file-id"
        Path(output).write_bytes(b"downloaded-image")
        return output

    monkeypatch.setattr(
        "dip_toolkit.assets.gdown.download",
        fake_download,
    )

    result = download_course_image(
        "cameraman_original.png",
        output_dir=tmp_path,
    )

    assert result == tmp_path / "cameraman_original.png"
    assert result.exists()


def test_rejects_unknown_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.json.return_value = {
        "ok": False,
        "error": "Arquivo não encontrado: unknown.png",
    }

    monkeypatch.setattr(
        "dip_toolkit.assets.requests.get",
        Mock(return_value=response),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Arquivo não encontrado",
    ):
        download_course_image(
            "unknown.png",
            output_dir=tmp_path,
        )
