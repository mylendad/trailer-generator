"""Telegram bot application: URL -> trailer, intro/outro management, rebuilds."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, StateFilter
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from magic_filter import F

from trailer.analysis.llm.provider import create_llm_client
from trailer.application.pipeline import Pipeline
from trailer.bot.media_templates import MediaTemplates
from trailer.config import Settings
from trailer.domain.exceptions import PipelineError
from trailer.observability.tracing import flush_tracing
from trailer.sources.youtube import YouTubeSource
from trailer.transcription.youtube_subtitles import YouTubeSubtitlesProvider

logger = logging.getLogger(__name__)

_YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?\S*|shorts/\S*)|youtu\.be/\S+)",
    re.IGNORECASE,
)

_REBUILD_TEMPERATURE = 0.9


def extract_youtube_url(text: str) -> str | None:
    """Return the first YouTube URL in the text, normalized with a scheme."""
    match = _YOUTUBE_RE.search(text)
    if not match:
        return None
    url = match.group(0)
    return url if url.lower().startswith("http") else f"https://{url}"


class _MenuCallback(CallbackData, prefix="menu"):
    action: str
    kind: str = "intro"
    name: str = ""


class _UploadStates(StatesGroup):
    waiting_intro = State()
    waiting_outro = State()


class BotService:
    """Wires the aiogram router to the pipeline and template storage."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._intro_templates = MediaTemplates(
            settings.telegram.intro_templates_dir, settings.telegram.active_intro_dir
        )
        self._outro_templates = MediaTemplates(
            settings.telegram.outro_templates_dir, settings.telegram.active_outro_dir
        )
        self._busy = asyncio.Lock()
        self._last_urls: dict[int, str] = {}
        self._variants: dict[int, int] = {}

    # ------------------------------------------------------------------ router

    def router(self) -> Router:
        router = Router(name="trailer")
        router.message.register(self._on_intro_file, StateFilter(_UploadStates.waiting_intro))
        router.message.register(self._on_outro_file, StateFilter(_UploadStates.waiting_outro))
        router.callback_query.register(self._on_upload, _MenuCallback.filter(F.action == "upload"))
        router.callback_query.register(self._on_select, _MenuCallback.filter(F.action == "select"))
        router.callback_query.register(self._on_pick, _MenuCallback.filter(F.action == "pick"))
        router.callback_query.register(
            self._on_rebuild, _MenuCallback.filter(F.action == "rebuild")
        )
        router.message.register(self._on_start, CommandStart())
        router.message.register(self._on_url, StateFilter(None))
        router.message.register(self._on_unknown)
        return router

    # ---------------------------------------------------------------- handlers

    async def _on_start(self, message: Message, state: FSMContext) -> None:
        await state.clear()
        await message.answer(
            "Пришлите ссылку YouTube — соберу содержательный трейлер "
            f"({self._pipeline_time_hint()}).\n\n{self._media_note()}",
            reply_markup=self._menu_markup().as_markup(),
        )

    async def _on_upload(
        self, query: CallbackQuery, callback_data: _MenuCallback, state: FSMContext
    ) -> None:
        if not isinstance(query.message, Message):
            return
        await query.answer()
        target = (
            _UploadStates.waiting_intro
            if callback_data.kind == "intro"
            else _UploadStates.waiting_outro
        )
        await state.set_state(target)
        label = "интро" if callback_data.kind == "intro" else "аутро"
        await query.message.answer(
            f"Пришлите {label} видеофайлом (mp4, до "
            f"{self._settings.telegram.max_intro_mb:.0f} МБ) — сохраню в шаблоны."
        )

    async def _on_select(self, query: CallbackQuery, callback_data: _MenuCallback) -> None:
        if not isinstance(query.message, Message):
            return
        await query.answer()
        templates = self._templates_for(callback_data.kind)
        names = templates.list()
        if not names:
            await query.message.answer(
                f"Шаблонов {'интро' if callback_data.kind == 'intro' else 'аутро'} "
                "пока нет. Сначала пришлите файл."
            )
            return
        builder = InlineKeyboardBuilder()
        for name in names:
            builder.button(
                text=name,
                callback_data=_MenuCallback(action="pick", kind=callback_data.kind, name=name),
            )
        builder.adjust(1)
        await query.message.answer("Выберите шаблон:", reply_markup=builder.as_markup())

    async def _on_pick(self, query: CallbackQuery, callback_data: _MenuCallback) -> None:
        if not isinstance(query.message, Message):
            return
        await query.answer()
        try:
            active = self._templates_for(callback_data.kind).activate(callback_data.name)
        except PipelineError as exc:
            await query.message.answer(str(exc))
            return
        label = "Интро" if callback_data.kind == "intro" else "Аутро"
        await query.message.answer(
            f"{label} выбрано: <b>{active.name}</b> — используется при следующей сборке.",
            reply_markup=self._menu_markup().as_markup(),
        )

    async def _on_rebuild(self, query: CallbackQuery, callback_data: _MenuCallback) -> None:
        if not isinstance(query.message, Message):
            return
        await query.answer()
        chat_id = query.message.chat.id
        url = self._last_urls.get(chat_id)
        if url is None:
            await query.message.answer("Сначала пришлите ссылку YouTube.")
            return
        if self._busy.locked():
            await query.message.answer("Уже идёт сборка — дождитесь окончания.")
            return
        await query.message.answer("🔄 Пересобираю с другими фрагментами…")
        asyncio.create_task(self._run_pipeline(message=query.message, url=url, rebuild=True))

    async def _on_intro_file(self, message: Message, state: FSMContext, bot: Bot) -> None:
        await self._save_media(message, bot, state, self._intro_templates, "intro.mp4", "интро")

    async def _on_outro_file(self, message: Message, state: FSMContext, bot: Bot) -> None:
        await self._save_media(message, bot, state, self._outro_templates, "outro.mp4", "аутро")

    async def _on_url(self, message: Message) -> None:
        url = extract_youtube_url(message.text or "")
        if url is None:
            return
        if self._busy.locked():
            await message.answer("Уже идёт сборка трейлера — дождитесь окончания.")
            return
        self._last_urls[message.chat.id] = url
        self._variants[message.chat.id] = 0
        await message.answer(f"⏳ Собираю трейлер… {self._pipeline_time_hint()}")
        asyncio.create_task(self._run_pipeline(message=message, url=url))

    async def _on_unknown(self, message: Message) -> None:
        if extract_youtube_url(message.text or "") is not None:
            return
        await message.answer(
            "Пришлите ссылку YouTube или воспользуйтесь кнопками.",
            reply_markup=self._menu_markup().as_markup(),
        )

    # ----------------------------------------------------------------- helpers

    def _templates_for(self, kind: str) -> MediaTemplates:
        return self._intro_templates if kind == "intro" else self._outro_templates

    @property
    def _max_intro_bytes(self) -> float:
        return self._settings.telegram.max_intro_mb * 1_000_000

    def _pipeline_time_hint(self) -> str:
        return "обычно это занимает 5–15 минут"

    def _media_note(self) -> str:
        intro = self._intro_templates.active()
        outro = self._outro_templates.active()
        parts = []
        parts.append(f"Интро: <b>{intro.name}</b>." if intro else "Интро не выбрано.")
        parts.append(f"Аутро: <b>{outro.name}</b>." if outro else "Аутро не выбрано.")
        return "\n".join(parts)

    def _menu_markup(self) -> InlineKeyboardBuilder:
        builder = InlineKeyboardBuilder()
        builder.button(
            text="📥 Прислать новое интро",
            callback_data=_MenuCallback(action="upload", kind="intro"),
        )
        builder.button(
            text="🎞 Выбрать интро", callback_data=_MenuCallback(action="select", kind="intro")
        )
        builder.button(
            text="📤 Прислать новое аутро",
            callback_data=_MenuCallback(action="upload", kind="outro"),
        )
        builder.button(
            text="🏁 Выбрать аутро", callback_data=_MenuCallback(action="select", kind="outro")
        )
        builder.button(text="🔄 Пересобрать", callback_data=_MenuCallback(action="rebuild"))
        builder.adjust(2, 2, 1)
        return builder

    def _variant_settings(self) -> Settings:
        """Settings copy with a hotter LLM so rebuilds pick other fragments."""
        settings = self._settings.model_copy(deep=True)
        settings.llm = settings.llm.model_copy(update={"temperature": _REBUILD_TEMPERATURE})
        return settings

    def _intro_attachment(self, message: Message) -> tuple[object | None, str | None]:
        if message.document:
            return message.document, message.document.file_name
        if message.video:
            return message.video, message.video.file_name
        return None, None

    async def _save_media(
        self,
        message: Message,
        bot: Bot,
        state: FSMContext,
        templates: MediaTemplates,
        fallback_name: str,
        label: str,
    ) -> None:
        attachment, name = self._intro_attachment(message)
        if attachment is None or not hasattr(attachment, "file_id"):
            await message.answer("Это не видео. Пришлите mp4 документом или видео.")
            return
        file_size = getattr(attachment, "file_size", None) or 0
        if file_size > self._max_intro_bytes:
            await message.answer(
                f"Файл больше {self._settings.telegram.max_intro_mb:.0f} МБ — "
                "Bot API не умеет скачивать такие файлы."
            )
            return
        temp_dir = self._settings.telegram.intro_templates_dir.parent / "tmp_bot_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / Path(name or fallback_name).name
        try:
            file = await bot.get_file(attachment.file_id)  # type: ignore[attr-defined]
            if file.file_path is None:
                await message.answer("Telegram не отдал путь к файлу, попробуйте снова.")
                return
            await bot.download_file(file.file_path, temp_file)
            saved = templates.save(name or fallback_name, temp_file)
        except PipelineError as exc:
            await message.answer(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - report download errors to the user
            logger.exception("Не удалось скачать %s", label)
            await message.answer(f"Не удалось сохранить файл: {exc}")
            return
        finally:
            temp_file.unlink(missing_ok=True)
        await state.clear()
        await message.answer(
            f"Шаблон {label} сохранён: <b>{saved.name}</b>.",
            reply_markup=self._menu_markup().as_markup(),
        )

    async def _run_pipeline(self, message: Message, url: str, *, rebuild: bool = False) -> None:
        settings = self._variant_settings() if rebuild else self._settings
        pipeline = Pipeline(
            settings=settings,
            source=YouTubeSource(settings.download),
            transcripts=YouTubeSubtitlesProvider(),
            llm=create_llm_client(settings.llm),
        )
        async with self._busy:
            try:
                trailer = await asyncio.to_thread(pipeline.run, url, "plan" if rebuild else None)
                await self._send_trailer(message, pipeline, trailer, rebuild)
            except PipelineError as exc:
                logger.error("Ошибка конвейера: %s", exc)
                await message.answer(f"Ошибка: {exc}")
            except Exception as exc:  # noqa: BLE001 - unexpected pipeline crash
                logger.exception("Необработанная ошибка конвейера")
                await message.answer(f"Неожиданная ошибка: {exc}")
            finally:
                flush_tracing()

    async def _send_trailer(
        self, message: Message, pipeline: Pipeline, trailer: Path, rebuild: bool
    ) -> None:
        chat_id = message.chat.id
        self._variants[chat_id] = self._variants.get(chat_id, 0) + 1
        variant = self._variants[chat_id]
        caption = "Трейлер готов." if not rebuild else f"Вариант {variant} готов."
        if pipeline.last_totals is not None:
            caption += f"\nТокены: {pipeline.last_totals.summary()}"
        video = FSInputFile(trailer)
        await message.answer_video(
            video=video,
            caption=caption,
            supports_streaming=True,
        )


def create_app(settings: Settings) -> tuple[Bot, Dispatcher]:
    token = settings.telegram.bot_token
    if token is None:
        raise SystemExit("Не задан TRAILER_TELEGRAM__BOT_TOKEN (токен бота из @BotFather)")
    bot = Bot(
        token=token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher()
    dispatcher.include_router(BotService(settings).router())
    return bot, dispatcher


async def main() -> None:
    settings = Settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot, dispatcher = create_app(settings)
    await dispatcher.start_polling(bot)
