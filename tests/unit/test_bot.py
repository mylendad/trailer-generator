from __future__ import annotations

from pathlib import Path

import pytest

from trailer.bot.app import BotService, extract_youtube_url
from trailer.bot.media_templates import MediaTemplates
from trailer.config import Settings
from trailer.domain.exceptions import EditingError


class TestExtractYoutubeUrl:
    def test_full_watch_url(self):
        url = "https://www.youtube.com/watch?v=abc123"
        assert extract_youtube_url(f"смотри: {url}") == url

    def test_short_url_without_scheme(self):
        assert extract_youtube_url("youtu.be/dQw4w9WgXcQ") == "https://youtu.be/dQw4w9WgXcQ"

    def test_shorts_url(self):
        url = "https://youtube.com/shorts/xyz?si=1"
        assert extract_youtube_url(url) == url

    def test_no_url(self):
        assert extract_youtube_url("просто текст") is None
        assert extract_youtube_url("") is None


class TestMediaTemplates:
    def _templates(self, tmp_path: Path) -> MediaTemplates:
        return MediaTemplates(tmp_path / "templates", tmp_path / "active")

    def test_save_and_list(self, tmp_path: Path):
        templates = self._templates(tmp_path)
        source = tmp_path / "source.mp4"
        source.write_bytes(b"data")
        saved = templates.save("мое интро v1!!", source)
        assert saved.name == "мое_интро_v1.mp4"
        assert templates.list() == ["мое_интро_v1"]

    def test_save_rejects_non_video(self, tmp_path: Path):
        templates = self._templates(tmp_path)
        source = tmp_path / "source.txt"
        source.write_text("nope")
        with pytest.raises(EditingError, match="Неподдерживаемый формат"):
            templates.save("intro", source)

    def test_activate_replaces_active(self, tmp_path: Path):
        templates = self._templates(tmp_path)
        for name in ("a", "b"):
            source = tmp_path / f"{name}.mp4"
            source.write_bytes(name.encode())
            templates.save(name, source)
        templates.activate("a")
        active = templates.active()
        assert active is not None and active.name == "a.mp4"
        templates.activate("b")
        active = templates.active()
        assert active is not None and active.name == "b.mp4"

    def test_activate_missing_template(self, tmp_path: Path):
        templates = self._templates(tmp_path)
        with pytest.raises(EditingError, match="Шаблон не найден"):
            templates.activate("ghost")

    def test_empty_state(self, tmp_path: Path):
        templates = self._templates(tmp_path)
        assert templates.list() == []
        assert templates.active() is None


class TestVariantSettings:
    def test_rebuild_boosts_temperature_without_mutating_original(self, tmp_path: Path):
        settings = Settings(workspace_dir=tmp_path)
        service = BotService(settings)
        variant = service._variant_settings()
        assert variant.llm.temperature == pytest.approx(0.9)
        assert settings.llm.temperature == pytest.approx(0.2)
