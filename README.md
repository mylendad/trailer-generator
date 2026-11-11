# AI Content Trailer Generator

Локальный Python-конвейер для автоматического создания содержательных видеотрейлеров
из длинных образовательных видео, лекций и выступлений.

Конвейер принимает ссылку на YouTube-видео, анализирует содержание речи с помощью LLM,
самостоятельно выбирает наиболее ценные смысловые фрагменты и склеивает из них
короткий MP4-трейлер (по умолчанию ~90 секунд).

## Возможности (MVP)

- Загрузка видео с YouTube (`yt-dlp`).
- Получение русскоязычных субтитров с таймкодами (`youtube-transcript-api`).
- Очистка и структурирование транскрипции, разбиение на смысловые блоки.
- Смысловой анализ блоков LLM: темы, идеи, аргументы, примеры, кандидаты.
- Построение общей карты содержания выступления.
- Детерминированный отбор и дедупликация кандидатов.
- Формирование сценария трейлера (зацепка → контекст → идеи → пример → вывод).
- Валидация сценария по исходной транскрипции (таймкоды, дубли, длительность).
- Нарезка и склейка FFmpeg без изменения композиции исходного видео.
- Сохранение всех промежуточных результатов и повторный запуск отдельных этапов.

## Установка

Требуется Python 3.12+, FFmpeg (в PATH), [mise](https://mise.jdx.dev/) (опционально).

```bash
mise install          # или вручную: убедиться, что python 3.12 и uv доступны
uv sync --group dev   # создать .venv и установить зависимости
```

Скопируйте `.env.example` в `.env` и укажите API-ключ LLM:

```bash
cp .env.example .env
```

## Использование

Запуск через `uv run` (venv активировать не нужно):

```bash
uv run python -m trailer.cli \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --duration 90 \
    --language ru
```

Либо через mise-задачу:

```bash
mise run cli --url "https://www.youtube.com/watch?v=VIDEO_ID" --duration 90
```

Либо напрямую из venv:

```bash
.venv/bin/python -m trailer.cli --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

Повторный запуск отдельных этапов без повторной загрузки видео:

```bash
# пересобрать сценарий и трейлер, используя сохранённые данные анализа
uv run python -m trailer.cli --url "..." --from-stage plan

# выполнить только один этап
uv run python -m trailer.cli --url "..." --only-stage validate
```

## Структура рабочей директории

```
workspace/
└── video_<video_id>/
    ├── video_info.json
    ├── source/original.mp4
    ├── transcript/{raw.json, prepared.json}
    ├── analysis/{blocks.json, block_analysis.json, content_map.json, candidates.json}
    ├── trailer/{script.json, script_validated.json, validation_report.json, trailer.mp4}
    └── logs/pipeline.log
```

## Конфигурация

Настройки задаются переменными окружения с префиксом `TRAILER_`
(см. `.env.example`) или аргументами CLI. Основные параметры:
язык субтитров, целевая длительность и допуск, рабочая директория,
LLM-провайдер/модель/ключ, размер смыслового блока, параметры загрузки.

### Интро и переходы

Если в `TRAILER_EDITING__INTRO_DIR` (по умолчанию `workspace/intro`) лежит mp4,
он добавляется в начало трейлера: приводится к формату исходника
(разрешение, fps, кодеки) и не учитывается в целевой длительности.
Между фрагментами делаются crossfade-переходы длительностью
`TRAILER_EDITING__TRANSITION_DURATION` (по умолчанию 0.5 с; 0 — без переходов).

Поддерживаются два LLM-провайдера (`TRAILER_LLM__PROVIDER`):

- `openai` — любой OpenAI-совместимый API (`TRAILER_LLM__API_KEY`, опционально `TRAILER_LLM__BASE_URL`);
- `gigachat` — Sber GigaChat. В `TRAILER_LLM__API_KEY` указывается Authorization key
  (base64 от `client_id:client_secret`) из личного кабинета GigaChat; одноразовый
  access token тоже принимается, но живёт ~30 минут. Модель по умолчанию `GigaChat-2-Max`.
  Проверка TLS-сертификата по умолчанию отключена (у GigaChat нет сертификата в
  стандартных хранилищах); если российский корневой сертификат установлен, задайте
  `TRAILER_LLM__VERIFY_SSL_CERTS=true` или путь к нему через
  `TRAILER_LLM__CA_BUNDLE_FILE`.

## Трассировка Langfuse (опционально)

Конвейер умеет отправлять трейсы в [Langfuse](https://langfuse.com): один трейс
`trailer.pipeline` на запуск, вложенные спаны этапов (`pipeline.*`), генерации
LLM (`llm.chat.completions`, `llm.gigachat.*`) с промптами, ответами, моделью,
длительностью и токенами (`prompt_tokens` / `completion_tokens` / `total_tokens`),
а также спан рендера с параметрами сегментов и размером файла.

Трассировка выключается автоматически, если ключи не заданы. Переменные
(стандартные для Langfuse, читаются из окружения или `.env`):

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000   # для self-hosted
```

Например, локальный запуск Langfuse:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse && docker compose up -d
```

## Разработка

```bash
mise run check        # lint + typecheck + tests
# или напрямую:
ruff check .
pyright
python -m pytest
```

## Архитектура

Код разделён на независимые компоненты (`src/trailer/`):

- `domain/` — Pydantic-модели данных и исключения;
- `sources/` — источники видео (интерфейс `VideoSource`, реализация YouTube);
- `transcription/` — источники транскрипции (интерфейс `TranscriptProvider`);
- `analysis/` — подготовка транскрипции, сегментация, LLM-анализ, отбор, планирование;
- `analysis/llm/` — изоляция LLM-провайдера за интерфейсом `LLMClient`;
- `validation/` — проверка сценария по исходным данным;
- `editing/` — FFmpeg-видеоредактор;
- `application/` — оркестрация этапов;
- `observability/` — опциональная Langfuse-трассировка (no-op без ключей);
- `cli.py` — интерфейс запуска.

Промпты хранятся отдельно от кода в `prompts/`.
