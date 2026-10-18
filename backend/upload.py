"""Повторяемая индексация: проверяем каждую книгу и сохраняем готовые записи."""

import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from numbers import Real
from threading import Lock

from qdrant_client import models

from .catalog import (
    book_id,
    book_key,
    prepare_catalog,
    source_fingerprint,
    validate_catalog,
)
from .storage import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    VECTOR_SIZE,
    create_client,
)

logger = logging.getLogger(__name__)
# Сериализуем загрузки внутри одного процесса Flask.
IMPORT_LOCK = Lock()
LLM_MODEL = "deepseek-flash"
ENRICHMENT_MODEL = LLM_MODEL
DENSE_MODEL = "nvidia/nemotron-3-embed-1b:free"
ENRICHMENT_PROMPT = """
Ты — книжный эксперт. Расширь краткое описание книги для поисковой системы.
Данные книги:
Название: {title}
Автор: {author}
Категория: {category}
Год: {year}
Исходное описание: {description}

Если книга хорошо известна, напиши аннотацию из 4–5 предложений:
ключевые темы, сюжетные мотивы и атмосферу, без спойлеров финала.
Если книга неизвестна, не придумывай сюжет: верни исходное описание.
Выдай только текст аннотации.
"""
# Изменение модели или промпта автоматически делает старое обогащение устаревшим.
ENRICHMENT_VERSION = hashlib.sha256(
    (ENRICHMENT_MODEL + ENRICHMENT_PROMPT + ":deepseek:thinking-disabled").encode("utf-8")
).hexdigest()
# При изменении правил построения текста увеличиваем TEXT_VERSION.
TEXT_VERSION = "book-text-v2"
DENSE_VERSION = f"openrouter:{DENSE_MODEL}:{VECTOR_SIZE}:{TEXT_VERSION}:input-type-passage-v1"

STAGE_ERRORS = {
    "lookup": "Не удалось проверить существующую запись.",
    "enrichment": "Не удалось получить обогащённое описание.",
    "dense": "Не удалось получить dense-вектор.",
    "save": "Не удалось подтвердить сохранение. Повторите загрузку.",
    "deduplication": "Книга подготовлена, но удаление дублей не подтверждено. Повторите загрузку.",
}


class IndexingError(ValueError):
    """Понятная пользователю ошибка конкретной книги."""


@lru_cache(maxsize=1)
def get_qdrant_client():
    return create_client()


@lru_cache(maxsize=1)
def get_llm_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(dotenv_path=".env", encoding="utf-8-sig")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY не найден")
    return OpenAI(
        base_url="https://api.deepseek.com",
        api_key=api_key, timeout=60, max_retries=2,
    )


@lru_cache(maxsize=1)
def get_embedding_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(dotenv_path=".env", encoding="utf-8-sig")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY не найден: он нужен для эмбеддингов")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key, timeout=60, max_retries=2,
    )


def enrich_description(book):
    """Ошибка API прерывает обработку книги; незаметной подмены описания нет."""
    result = get_llm_client().chat.completions.create(
        model=ENRICHMENT_MODEL,
        messages=[{"role": "user", "content": ENRICHMENT_PROMPT.format(**book)}],
        extra_body={"thinking": {"type": "disabled"}},
    )
    return result.choices[0].message.content


def get_dense_vector(text, *, input_type="passage"):
    """Книги кодируем как passage, поисковые запросы — как query."""
    if input_type not in {"passage", "query"}:
        raise ValueError("input_type должен быть passage или query")
    # NVIDIA добавляет префикс по input_type; вручную его в текст не вставляем.
    # Используем родную размерность 2048 без сокращения вектора.
    result = get_embedding_client().embeddings.create(
        model=DENSE_MODEL, input=[text], encoding_format="float",
        extra_body={"input_type": input_type},
    )
    return result.data[0].embedding


def valid_dense(vector):
    if not isinstance(vector, list) or len(vector) != VECTOR_SIZE:
        return False
    if not all(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
        for value in vector
    ):
        return False
    norm = math.hypot(*vector)
    return math.isfinite(norm) and norm > 0


def render_index_text(book, enriched):
    # Регистр названия/автора не меняет ни идентичность, ни текст для векторизации.
    title, author = book_key(book["title"], book["author"])
    return (
        f"Название: {title}. Автор: {author}. Жанр: {book['category']}. "
        f"Год: {book['year']}. Описание: {enriched}"
    )


def get_existing_books():
    """Читаем метаданные всех страниц; векторы запрашиваем только нужным книгам."""
    existing = {}
    offset = None
    while True:
        points, offset = get_qdrant_client().scroll(
            collection_name=COLLECTION_NAME, limit=256, offset=offset,
            with_payload=["title", "author"], with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            title, author = payload.get("title"), payload.get("author")
            if isinstance(title, str) and isinstance(author, str):
                existing.setdefault(book_key(title, author), []).append(point.id)
        if offset is None:
            return existing


def parse_created_at(value):
    """Для сравнения дат нужен ISO 8601 с часовым поясом; старые даты неизвестны."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        pass
    return None


def read_existing(book, existing, client):
    ids = existing.get(book_key(book["title"], book["author"]), [])
    # Проверяем вычисленный ID даже при отсутствии книги в снимке каталога.
    point_ids = ids or [book_id(book)]
    records = client.retrieve(
        COLLECTION_NAME, ids=point_ids, with_payload=True, with_vectors=[DENSE_VECTOR_NAME],
    )
    if ids and {str(record.id) for record in records} != {str(point_id) for point_id in ids}:
        raise IndexingError("Запись изменилась во время загрузки. Повторите загрузку.")
    if not records:
        return None
    for record in records:
        payload = record.payload or {}
        title, author = payload.get("title"), payload.get("author")
        if (
            not isinstance(title, str) or not isinstance(author, str)
            or book_key(title, author) != book_key(book["title"], book["author"])
        ):
            raise IndexingError("ID занят другой книгой. Запись сохранена без изменений.")

    def order(record):
        created_at = parse_created_at((record.payload or {}).get("created_at"))
        # При неизвестных или одинаковых датах сначала предпочитаем постоянный UUID.
        # Остальные ID задают стабильный выбор, но не говорят о времени создания.
        return (
            created_at or datetime.min.replace(tzinfo=timezone.utc),
            str(record.id) == book_id(book), str(record.id),
        )

    return max(records, key=order)


def remove_duplicates(book, record, existing, client):
    """Удаляем лишние копии только после успешной обработки выбранной записи."""
    if record is None:
        return 0
    ids = existing.get(book_key(book["title"], book["author"]), [])
    duplicate_ids = [point_id for point_id in ids if str(point_id) != str(record.id)]
    if not duplicate_ids:
        return 0
    result = client.delete(
        COLLECTION_NAME, points_selector=models.PointIdsList(points=duplicate_ids), wait=True,
    )
    if result.status != models.UpdateStatus.COMPLETED:
        raise IndexingError(STAGE_ERRORS["deduplication"])
    return len(duplicate_ids)


def payload_fingerprint(payload):
    try:
        return source_fingerprint(validate_catalog([payload])[0])
    except ValueError:
        return None


def process_book(book, record, client, progress):
    payload = (record.payload or {}) if record is not None else {}
    metadata = payload.get("indexing", {})
    if not isinstance(metadata, dict):
        metadata = {}
    fingerprint = source_fingerprint(book)
    same_source = (
        metadata.get("source_hash") == fingerprint
        and payload_fingerprint(payload) == fingerprint
    )
    enriched = payload.get("enriched_description")
    reuse_enrichment = (
        same_source and metadata.get("enrichment_version") == ENRICHMENT_VERSION
        and isinstance(enriched, str) and bool(enriched.strip())
    )
    if not reuse_enrichment:
        progress["stage"] = "enrichment"
        enriched = enrich_description(book)
        if not isinstance(enriched, str) or not enriched.strip():
            raise IndexingError("Модель вернула пустое обогащённое описание.")
        enriched = enriched.strip()

    text = render_index_text(book, enriched)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    same_text = reuse_enrichment and metadata.get("text_hash") == text_hash
    vectors = record.vector if record is not None and isinstance(record.vector, dict) else {}
    dense = vectors.get(DENSE_VECTOR_NAME)
    reuse_dense = (
        same_text and metadata.get("dense_version") == DENSE_VERSION and valid_dense(dense)
    )
    if reuse_dense:
        return "skipped"

    progress["stage"] = "dense"
    dense = get_dense_vector(text)
    if not valid_dense(dense):
        raise IndexingError("Получен некорректный dense-вектор: проверьте размерность и значения.")
    saved_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    point = models.PointStruct(
        id=record.id if record is not None else book_id(book),
        vector={DENSE_VECTOR_NAME: dense},
        payload={
            **payload, **book,
            "created_at": payload.get("created_at") if record is not None else saved_at,
            "updated_at": saved_at,
            "enriched_description": enriched,
            "indexing": {
                "source_hash": fingerprint,
                "enrichment_version": ENRICHMENT_VERSION,
                "dense_version": DENSE_VERSION,
                "text_hash": text_hash,
            },
        },
    )
    progress["stage"] = "save"
    # Старую запись заменяем только после успешной подготовки dense-вектора.
    result = client.upsert(COLLECTION_NAME, [point], wait=True)
    if result.status != models.UpdateStatus.COMPLETED:
        raise IndexingError(STAGE_ERRORS["save"])
    if record is None:
        return "added"
    if (
        not same_source or not reuse_enrichment
        or metadata.get("dense_version") != DENSE_VERSION
    ):
        return "updated"
    return "repaired"


def add_books_from_json(file_path):
    """Проверяем весь файл, затем независимо обрабатываем и сохраняем книги."""
    with open(file_path, "r", encoding="utf-8") as file:
        books, file_stats = prepare_catalog(json.load(file))

    report = {
        **file_stats,
        "total": len(books), "added": 0, "updated": 0, "repaired": 0,
        "skipped": 0, "failed": 0, "duplicates_removed": 0, "errors": [],
    }
    with IMPORT_LOCK:
        client = get_qdrant_client()
        existing = get_existing_books()
        for book in books:
            progress = {"stage": "lookup"}
            try:
                record = read_existing(book, existing, client)
                outcome = process_book(book, record, client, progress)
                progress["stage"] = "deduplication"
                report["duplicates_removed"] += remove_duplicates(book, record, existing, client)
                report[outcome] += 1
            except Exception as error:
                logger.exception("Ошибка книги %s на этапе %s", book["title"], progress["stage"])
                report["failed"] += 1
                report["errors"].append({
                    "title": book["title"], "author": book["author"],
                    "stage": progress["stage"],
                    "error": str(error) if isinstance(error, IndexingError)
                    else STAGE_ERRORS[progress["stage"]],
                })
    report["status"] = (
        "failed" if report["failed"] == report["total"]
        else "partial" if report["failed"] else "success"
    )
    report["message"] = (
        f"Добавлено: {report['added']}; обновлено: {report['updated']}; "
        f"восстановлено: {report['repaired']}; пропущено: {report['skipped']}; "
        f"ошибок: {report['failed']}."
        f" В файле одинаковых повторов: {report['file_duplicates']}; "
        f"замен с изменениями: {report['file_conflicts']} (взята последняя запись)."
        f" Удалено дублей из базы: {report['duplicates_removed']}."
    )
    return report
