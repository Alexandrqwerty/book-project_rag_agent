# Загрузка книг для агента Книжный эксперт

import json
import time
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from openai import OpenAI
from fastembed import SparseTextEmbedding
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", encoding="utf-8")

# Подключения
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY не найден")

CLOUDRU_API_KEY = os.getenv("CLOUDRU_API_KEY")
if not CLOUDRU_API_KEY:
    raise ValueError("CLOUDRU_API_KEY не найден")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
local_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

openai_client = OpenAI(
    base_url="https://foundation-models.api.cloud.ru/v1",
    api_key=CLOUDRU_API_KEY,
)

collection_name = "books_collection"
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

VECTOR_SIZE = 1536

def enrich_description(book):
    """Обогащает описание книги через LLM"""
    prompt = f"""
    Ты — профессиональный книжный эксперт. Твоя задача — расширить краткое описание книги для поисковой системы.
    Данные книги:
    Название: {book.get('title', '')}
    Автор: {book.get('author', '')}
    Категория: {book.get('category', '')}
    Текущее описание: {book.get('description', '')}

    Инструкция:
    1. Если книга тебе хорошо известна, напиши развернутую аннотацию (4-5 предложений). Опиши ключевые темы, скрытые смыслы, тропы сюжета и атмосферу, не раскрывая главных спойлеров финала.
    2. Если книга тебе НЕ известна, НЕ придумывай сюжет. Просто верни исходное описание.

    Выдай ТОЛЬКО текст новой аннотации, без вводных слов и приветствий.
    """
    try:
        completion = openai_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            timeout=5
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️ Ошибка LLM: {e}, используем оригинал")
        return book.get('description', '')

def get_dense_vector(text):
    """Генерирует плотный вектор через OpenRouter"""
    try:
        res = openai_client.embeddings.create(
            model="openai/text-embedding-3-small",
            input=[text[:8000]]
        )
        return res.data[0].embedding
    except Exception as e:
        print(f"   ⚠️ Ошибка эмбеддинга: {e}")
        return [0.0] * VECTOR_SIZE

def get_existing_books():
    """Возвращает множество (title, author) уже существующих книг"""
    existing = set()
    scroll = local_client.scroll(collection_name, limit=10000, with_payload=True)
    for point in scroll[0]:
        title = point.payload.get('title', '')
        author = point.payload.get('author', '')
        if title and author:
            existing.add((title.lower(), author.lower()))
    return existing

def add_books_from_json(file_path):
    """Добавляет новые книги из JSON с обогащением описаний"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'books' in data:
        books = data['books']
    elif isinstance(data, list):
        books = data
    else:
        return 0, "Неизвестная структура JSON"
    existing_books = get_existing_books()
    new_books = []
    for book in books:
        title = book.get('title', '')
        author = book.get('author', '')
        key = (title.lower(), author.lower())
        if key not in existing_books:
            new_books.append(book)
    if not new_books:
        return 0, "Новых книг не найдено"
    count = local_client.count(collection_name).count
    points = []
    total = len(new_books)
    for idx, book in enumerate(new_books):
        print(f"[{idx+1}/{total}] Обработка: {book.get('title', '')}")
        enriched = enrich_description(book)
        time.sleep(2)
        text_for_embed = f"Название: {book.get('title', '')}. Автор: {book.get('author', '')}. Жанр: {book.get('category', '')}. Описание: {enriched}"
        dense_vec = get_dense_vector(text_for_embed)
        sparse_emb = next(sparse_model.embed([text_for_embed]))
        sparse_vec = {"indices": sparse_emb.indices.tolist(), "values": sparse_emb.values.tolist()}
        point = PointStruct(
            id=count + idx,
            vector={"dense-vector": dense_vec, "sparse-vector": sparse_vec},
            payload={
                "title": book.get('title', ''),
                "author": book.get('author', ''),
                "category": book.get('category', ''),
                "year": book.get('year', ''),
                "description": book.get('description', ''),
                "enriched_description": enriched,
            },
        )
        points.append(point)
        print("   ✅ Готово")
    if points:
        local_client.upsert(collection_name, points)
    return len(points), f"Добавлено {len(points)} новых книг с обогащёнными описаниями"
