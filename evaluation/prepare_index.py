"""Создать оценочный индекс из сохранённых описаний; обогащение не вызывается."""

import argparse
import hashlib

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from backend.catalog import source_fingerprint, validate_catalog
from backend.storage import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    VECTOR_SIZE,
    prepare_collection,
)
from backend.upload import (
    DENSE_MODEL,
    DENSE_VERSION,
    ENRICHMENT_VERSION,
    get_dense_vector,
    render_index_text,
    valid_dense,
)

from .run import Cache, load_dataset


def prepare_index(client, dataset, cache):
    # Никогда не очищаем и не заменяем существующую коллекцию.
    if client.collection_exists(COLLECTION_NAME):
        raise ValueError("Оценочная коллекция уже существует. Используйте её для прогона "
                         "или запустите новый пустой экземпляр Qdrant.")
    sources = {book["book_id"]: book for book in dataset["catalog"]}
    evidence = dataset.get("database_evidence_snapshot")
    if not evidence or len(evidence) != len(sources):
        raise ValueError("Нужен полный снимок обогащённых описаний")
    points = []
    for item in evidence:
        book = validate_catalog([sources[item["book_id"]]])[0]
        if source_fingerprint(item) != source_fingerprint(book):
            raise ValueError("Исходные поля снимка отличаются от каталога")
        enriched = item["enriched_description"]
        if not isinstance(enriched, str) or not enriched.strip():
            raise ValueError("Пустое описание снимка")
        text = render_index_text(book, enriched)
        vector, _ = cache.get("passage", {
            "model": DENSE_MODEL, "version": DENSE_VERSION, "dimensions": VECTOR_SIZE,
            "input_type": "passage", "text": text,
        }, lambda text=text: get_dense_vector(text, input_type="passage"))
        if not valid_dense(vector):
            raise ValueError("Некорректный вектор книги")
        points.append(models.PointStruct(
            id=item["book_id"], vector={DENSE_VECTOR_NAME: vector},
            payload={**book, "enriched_description": enriched, "indexing": {
                "source_hash": source_fingerprint(book),
                "enrichment_version": ENRICHMENT_VERSION, "dense_version": DENSE_VERSION,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }},
        ))
    # Создаём коллекцию только после подготовки всех векторов.
    prepare_collection(client)
    result = client.upsert(COLLECTION_NAME, points, wait=True)
    if result.status != models.UpdateStatus.COMPLETED:
        raise RuntimeError("Сохранение оценочного индекса не подтверждено")
    return len(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evaluation/book_search_v4.json")
    parser.add_argument("--port", type=int, default=6335)
    parser.add_argument("--cache-dir", default="evaluation/cache")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.port == 6333:
        parser.error("Нужен отдельный Qdrant: используйте порт 6335, не рабочий 6333")
    load_dotenv(".env", encoding="utf-8-sig")
    dataset, _ = load_dataset(args.dataset, 10000)
    with QdrantClient(host="127.0.0.1", port=args.port, timeout=30) as client:
        count = prepare_index(client, dataset, Cache(args.cache_dir, args.offline))
    print(f"Подготовлено {count} книг на порту {args.port}; обогащение взято из снимка.")


if __name__ == "__main__":
    main()
