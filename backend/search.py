"""Поиск по каталогу, необязательный реранкинг и ответ на основе найденных книг."""

import json
import math
import os
from numbers import Real

from .catalog import validate_catalog
from .prompts import SYSTEM_PROMPT
from .storage import COLLECTION_NAME, DENSE_VECTOR_NAME
from .upload import (
    LLM_MODEL,
    get_dense_vector,
    get_llm_client,
    get_qdrant_client,
    valid_dense,
)

ANSWER_MODEL = LLM_MODEL
RERANK_URL = "https://openrouter.ai/api/v1/rerank"
DEFAULT_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
TOP_K = 5
CANDIDATE_COUNT = 20


class SearchError(RuntimeError):
    """Ошибка поиска с безопасным для показа пользователю сообщением."""


def rerank_enabled():
    value = os.getenv("RERANK_ENABLED", "false").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError("RERANK_ENABLED должен быть true или false")
    return value == "true"


def validate_search_config():
    if rerank_enabled() and not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise ValueError("Для RERANK_ENABLED=true нужен OPENROUTER_API_KEY")
    if not os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL).strip():
        raise ValueError("RERANK_MODEL не должен быть пустым")


def search_books(query, limit=TOP_K):
    try:
        vector = get_dense_vector(query, input_type="query")
        if not valid_dense(vector):
            raise ValueError("Некорректный вектор запроса")
    except Exception as error:
        raise SearchError("Не удалось получить вектор запроса. Попробуйте позже.") from error

    return search_books_by_vector(vector, limit=limit)


def search_books_by_vector(vector, limit=TOP_K):
    """Поиск с уже готовым вектором: оценочные прогоны могут брать его из кеша."""
    if not valid_dense(vector):
        raise SearchError("Некорректный вектор запроса.")
    try:
        points = get_qdrant_client().query_points(
            collection_name=COLLECTION_NAME, query=vector, using=DENSE_VECTOR_NAME,
            limit=limit, with_payload=True,
        ).points
    except Exception as error:
        raise SearchError("База книг недоступна. Попробуйте позже.") from error

    books = []
    for point in points:
        try:
            payload = point.payload or {}
            book = validate_catalog([payload])[0]
        except ValueError as error:
            raise SearchError("В индексе есть некорректные записи. Повторите загрузку каталога.") from error
        enriched = payload.get("enriched_description")
        books.append({
            "id": str(point.id), **book,
            "enriched_description": enriched.strip()
            if isinstance(enriched, str) and enriched.strip() else book["description"],
            "dense_score": point.score,
        })
    return books


def rerank_books(query, books, limit=TOP_K):
    import requests

    if not books:
        return []
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SearchError("Для реранкинга не настроен ключ OpenRouter.")
    documents = [
        f"Название: {book['title']}. Автор: {book['author']}. "
        f"Жанр: {book['category']}. Год: {book['year']}. "
        f"Исходное описание: {book['description']}. "
        f"Обогащённое описание: {book['enriched_description']}"
        for book in books
    ]
    top_n = min(limit, len(books))
    try:
        response = requests.post(
            RERANK_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL),
                "query": query, "documents": documents, "top_n": top_n,
            },
            timeout=(5, 30),
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or len(results) != top_n:
            raise ValueError("Некорректное число результатов реранкера")
        ranked, seen = [], set()
        for result in results:
            index, score = result["index"], result["relevance_score"]
            if type(index) is not int or not 0 <= index < len(books) or index in seen:
                raise ValueError("Некорректный или повторный индекс реранкера")
            if not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(score):
                raise ValueError("Некорректная оценка реранкера")
            seen.add(index)
            ranked.append({**books[index], "rerank_score": score})
        return sorted(ranked, key=lambda book: book["rerank_score"], reverse=True)
    except Exception as error:
        # Включённый реранкер не подменяется незаметно обычным dense-поиском.
        raise SearchError("Реранкер недоступен или вернул некорректный ответ. Попробуйте позже.") from error


def retrieve_books(query):
    enabled = rerank_enabled()
    books = search_books(query, limit=CANDIDATE_COUNT if enabled else TOP_K)
    return rerank_books(query, books) if enabled and books else books


def render_answer(content, books):
    """Проверяем ID из ответа модели; названия и авторов берём из каталога."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Пустой ответ модели")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Ответ должен быть JSON-объектом")
    recommendations, summary = data.get("recommendations"), data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Нет объяснения результата")
    if not isinstance(recommendations, list) or len(recommendations) > min(TOP_K, len(books)):
        raise ValueError("Некорректный список рекомендаций")
    catalog = {book["id"]: book for book in books}
    seen, lines = set(), []
    for item in recommendations:
        if not isinstance(item, dict):
            raise ValueError("Рекомендация должна быть объектом")
        point_id, reason = item.get("id"), item.get("reason")
        if not isinstance(point_id, str) or point_id not in catalog or point_id in seen:
            raise ValueError("Модель указала неизвестную книгу или повторила ID")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Нет объяснения рекомендации")
        seen.add(point_id)
        book = catalog[point_id]
        lines.append(
            f"{len(lines) + 1}. {book['title']} — {book['author']} "
            f"({book['year']}, {book['category']})\n{reason.strip()}"
        )
    return "\n\n".join([*lines, summary.strip()])


def request_answer(query, books, *, thinking="disabled"):
    """Общий вызов LLM для приложения и оценки; сохраняет JSON и usage API."""
    if thinking not in {"enabled", "disabled"}:
        raise ValueError("Неизвестный режим thinking")
    context = [{key: book[key] for key in (
        "id", "title", "author", "category", "year", "description", "enriched_description",
    )} for book in books]
    return get_llm_client().chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"query": query, "books": context}, ensure_ascii=False,
            )},
        ],
        extra_body={"thinking": {"type": thinking}},
        response_format={"type": "json_object"},
    )


def generate_answer(query, books):
    try:
        response = request_answer(query, books)
        return render_answer(response.choices[0].message.content, books)
    except Exception as error:
        raise SearchError("Не удалось сформировать корректную рекомендацию. Попробуйте позже.") from error
