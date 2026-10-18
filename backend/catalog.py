"""Проверка каталога и постоянные идентификаторы произведений."""

import hashlib
import json
import unicodedata
from uuid import NAMESPACE_URL, uuid5

# Эта константа — часть правила формирования ID; менять её после загрузки нельзя.
BOOK_NAMESPACE = uuid5(
    NAMESPACE_URL, "https://github.com/Alexandrqwerty/book-project_rag_agent/books"
)
TEXT_FIELDS = ("title", "author", "category", "description")


class CatalogValidationError(ValueError):
    """Каталог не соответствует ожидаемой структуре книг."""


def clean_text(value: str) -> str:
    """Убираем лишние пробелы и выравниваем Unicode-представление текста."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def book_key(title: str, author: str) -> tuple[str, str]:
    """В этом каталоге произведение определяется названием и автором."""
    return clean_text(title).casefold(), clean_text(author).casefold()


def book_id(book: dict) -> str:
    # JSON сохраняет границу полей: пары ('a|b', 'c') и ('a', 'b|c') различаются.
    identity = json.dumps(book_key(book["title"], book["author"]), ensure_ascii=False)
    return str(uuid5(BOOK_NAMESPACE, identity))


def source_fingerprint(book: dict) -> str:
    """ID определяет произведение, а этот хеш — версию его исходных данных."""
    title, author = book_key(book["title"], book["author"])
    data = {
        "title": title, "author": author, "year": book["year"],
        "category": clean_text(book["category"]),
        "description": clean_text(book["description"]),
    }
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_catalog(data: object) -> list[dict]:
    """Проверяем каталог и возвращаем по последнему вхождению каждой книги."""
    books, _ = prepare_catalog(data)
    return books


def prepare_catalog(data: object) -> tuple[list[dict], dict]:
    """Проверяем все записи, выбираем последние версии и считаем повторы."""
    books = data.get("books") if isinstance(data, dict) else data
    if not isinstance(books, list):
        raise CatalogValidationError(
            'Ожидается список книг или объект {"books": [...]}'
        )
    if not books:
        raise CatalogValidationError("Каталог не должен быть пустым")

    unique = {}
    stats = {"input_total": len(books), "file_duplicates": 0, "file_conflicts": 0}
    for position, raw in enumerate(books, start=1):
        if not isinstance(raw, dict):
            raise CatalogValidationError(f"Книга №{position}: ожидается объект")
        book = {}
        for field in TEXT_FIELDS:
            value = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                raise CatalogValidationError(
                    f"Книга №{position}: поле {field} должно быть непустой строкой"
                )
            book[field] = clean_text(value)
        # Проверяем точный тип int, чтобы не принять True/False в поле year.
        if type(raw.get("year")) is not int:
            raise CatalogValidationError(
                f"Книга №{position}: поле year должно быть целым числом"
            )
        book["year"] = raw["year"]
        key = book_key(book["title"], book["author"])
        if key in unique:
            previous = unique[key]
            changed = any(book[field] != previous[field] for field in (
                "category", "description", "year"
            ))
            stats["file_conflicts" if changed else "file_duplicates"] += 1
        # Заменяем запись целиком: поля разных версий не смешиваем.
        unique[key] = book
    return list(unique.values()), stats
