"""Настройки коллекции и безопасная подготовка Qdrant при запуске."""

import logging
import os
import time

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

# Отдельная коллекция: векторы разных embedding-моделей несовместимы.
COLLECTION_NAME = "books_nemotron3_2048"
DENSE_VECTOR_NAME = "dense-vector"
VECTOR_SIZE = 2048

logger = logging.getLogger(__name__)


def create_client() -> QdrantClient:
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        timeout=3,
    )


def wait_for_qdrant(
    client: QdrantClient, attempts: int = 10, delay: float = 1.0
) -> None:
    """Ограниченно повторяем подключение при временной недоступности базы."""
    if attempts < 1 or delay < 0:
        raise ValueError("attempts должен быть >= 1, delay должен быть >= 0")

    for attempt in range(1, attempts + 1):
        try:
            client.get_collections()
            return
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            # Ошибки доступа и некорректные запросы ожиданием не исправить.
            if isinstance(exc, UnexpectedResponse) and exc.status_code not in {
                502, 503, 504
            }:
                raise
            if attempt == attempts:
                raise RuntimeError(
                    f"Qdrant недоступен после {attempts} попыток. "
                    "Проверьте запуск базы, QDRANT_HOST и QDRANT_PORT."
                ) from exc
            logger.warning("Ожидание Qdrant: попытка %s/%s", attempt, attempts)
            time.sleep(delay)


def validate_collection(client: QdrantClient) -> None:
    """Проверяем нужный dense-вектор; дополнительные старые векторы допустимы."""
    params = client.get_collection(COLLECTION_NAME).config.params
    dense = (
        params.vectors.get(DENSE_VECTOR_NAME)
        if isinstance(params.vectors, dict)
        else None
    )
    problems = []

    if dense is None:
        problems.append(f"отсутствует именованный вектор {DENSE_VECTOR_NAME}")
    else:
        if dense.size != VECTOR_SIZE:
            problems.append(f"размерность dense {dense.size}, ожидается {VECTOR_SIZE}")
        if dense.distance != models.Distance.COSINE:
            problems.append("метрика dense должна быть Cosine")
        if dense.multivector_config is not None:
            problems.append("ожидается один dense-вектор на книгу, не multivector")

    if problems:
        raise RuntimeError(
            f"Коллекция {COLLECTION_NAME} несовместима: {'; '.join(problems)}. "
            "Данные сохранены. Требуется отдельная миграция коллекции."
        )


def prepare_collection(client: QdrantClient) -> None:
    """Ждём базу, создаём отсутствующую коллекцию и проверяем её схему."""
    wait_for_qdrant(client)
    if not client.collection_exists(COLLECTION_NAME):
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=VECTOR_SIZE, distance=models.Distance.COSINE
                    )
                },
            )
        except UnexpectedResponse as exc:
            # Другой процесс мог создать коллекцию между проверкой и созданием.
            if exc.status_code != 409:
                raise

    validate_collection(client)
    logger.info("Коллекция %s готова", COLLECTION_NAME)
