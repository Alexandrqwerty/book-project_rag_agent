"""Формат каталога, одинаковые книги и постоянство ID."""

import unittest

from backend.catalog import (
    CatalogValidationError,
    book_id,
    source_fingerprint,
    validate_catalog,
)


def example_book(**changes):
    return {
        "title": "Война и мир",
        "author": "Лев Толстой",
        "category": "Роман",
        "description": "История нескольких семей в эпоху Наполеона.",
        "year": 1869,
        **changes,
    }


class CatalogTests(unittest.TestCase):
    def test_fingerprint_tracks_source_changes_but_ignores_identity_formatting(self):
        original = source_fingerprint(example_book())
        self.assertEqual(original, source_fingerprint(example_book(title=" ВОЙНА  И МИР ")))
        for change in ({"year": 1870}, {"description": "Новое"}, {"category": "История"}):
            self.assertNotEqual(original, source_fingerprint(example_book(**change)))

    def test_accepts_both_formats_and_normalizes_without_mutating_input(self):
        raw = example_book(title="  Война   и мир ")
        expected = example_book()
        self.assertEqual(validate_catalog([raw]), [expected])
        self.assertEqual(validate_catalog({"books": [raw]}), [expected])
        self.assertEqual(raw["title"], "  Война   и мир ")

    def test_invalid_structure_and_fields_have_useful_errors(self):
        cases = [
            (None, "Ожидается список"),
            ({"books": {}}, "Ожидается список"),
            ([], "не должен быть пустым"),
            (["book"], "Книга №1: ожидается объект"),
            ([example_book(), example_book(author="  ")], "Книга №2: поле author"),
            ([example_book(year=True)], "поле year"),
            ([example_book(year="1869")], "поле year"),
        ]
        for data, error in cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(CatalogValidationError, error):
                    validate_catalog(data)
        for field in ("title", "author", "category", "description", "year"):
            book = example_book()
            del book[field]
            with self.subTest(missing=field):
                with self.assertRaisesRegex(CatalogValidationError, field):
                    validate_catalog([book])

    def test_duplicates_are_collapsed_and_last_conflicting_entry_wins(self):
        book = example_book()
        duplicate = example_book(title=" ВОЙНА   И МИР ", author="ЛЕВ ТОЛСТОЙ")
        self.assertEqual(validate_catalog([book, duplicate]), [
            example_book(title="ВОЙНА И МИР", author="ЛЕВ ТОЛСТОЙ"),
        ])
        for field, value in (
            ("year", 1870), ("category", "История"), ("description", "Другое описание")
        ):
            with self.subTest(field=field):
                conflict = example_book(**{field: value})
                self.assertEqual(validate_catalog([book, conflict]), [conflict])

    def test_id_ignores_case_spacing_and_unicode_composition(self):
        self.assertEqual(
            book_id(example_book()),
            book_id(example_book(title=" ВОЙНА   И МИР ", author="ЛЕВ ТОЛСТОЙ")),
        )
        self.assertEqual(
            book_id(example_book(title="й")),
            book_id(example_book(title="и\u0306")),
        )

    def test_metadata_changes_keep_id_but_other_works_have_different_ids(self):
        original = book_id(example_book())
        self.assertEqual(original, book_id(example_book(year=1870, description="Новое")))
        self.assertNotEqual(original, book_id(example_book(author="Другой автор")))
        self.assertNotEqual(original, book_id(example_book(title="Другое название")))
        self.assertNotEqual(
            book_id(example_book(title="a|b", author="c")),
            book_id(example_book(title="a", author="b|c")),
        )


if __name__ == "__main__":
    unittest.main()
