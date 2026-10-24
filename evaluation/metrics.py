"""Метрики по согласованным оценкам; ошибки API сюда не передаются."""


def recall(ids, grades, minimum_grade=2):
    relevant = {key for key, grade in grades.items() if grade >= minimum_grade}
    return len(set(ids) & relevant) / len(relevant) if relevant else None


def score_answer(answer, query, dense_ids, context_ids):
    grades = {item["book_id"]: item["grade"] for item in query["judgments"]}
    if any(value is None for value in grades.values()):
        raise ValueError("Не все книги размечены")
    ids = [item["id"] for item in answer["recommendations"]]
    if len(ids) != len(set(ids)) or len(ids) > 5 or not set(ids) <= set(context_ids):
        raise ValueError("Ответ содержит неизвестные, повторные ID или больше пяти книг")
    if not set(context_ids) <= grades.keys() or not set(dense_ids) <= grades.keys():
        raise ValueError("Кандидаты отсутствуют в оценочном каталоге")
    strong = {key for key, grade in grades.items() if grade == 2}
    hits = len(set(ids) & strong)
    base = {
        "dense_recall_strong": recall(dense_ids, grades),
        "context_recall_strong": recall(context_ids, grades),
        "dense_recall_including_partial": recall(dense_ids, grades, 1),
        "context_recall_including_partial": recall(context_ids, grades, 1),
        "recommended_count": len(ids),
        "strong_count": hits,
        "partial_count": sum(grades[key] == 1 for key in ids),
        "irrelevant_count": sum(grades[key] == 0 for key in ids),
        # Текст объяснений требует отдельного чтения; здесь оцениваются ID.
        "precision_strong": None,
        # Только соответствие ID; оговорки для оценок 1 проверяются отдельно.
        "precision_supported": None,
        "coverage_capped_at_5": None,
        "false_refusal": None,
        "correct_no_match": None,
    }
    if query["expected_behavior"] == "no_match":
        if any(grades.values()):
            raise ValueError("no_match противоречит положительной разметке")
        base["correct_no_match"] = int(not ids)
    else:
        if not strong:
            raise ValueError("Для этой метрики нужен хотя бы один ответ с оценкой 2")
        # Пустой ответ на разрешимый запрос получает 0, а не исчезает из среднего.
        base["precision_strong"] = hits / len(ids) if ids else 0.0
        base["precision_supported"] = sum(grades[key] >= 1 for key in ids) / len(ids) if ids else 0.0
        # Это наша метрика с ограничением ответа, не стандартный Recall@5.
        base["coverage_capped_at_5"] = hits / min(5, len(strong))
        base["false_refusal"] = int(not ids)
    return base


def average_metrics(rows):
    successful = [row for row in rows if row["status"] == "success"]
    keys = (
        "dense_recall_strong", "context_recall_strong", "precision_strong",
        "precision_supported",
        "coverage_capped_at_5", "false_refusal", "correct_no_match",
        "partial_count", "irrelevant_count",
    )
    result = {"attempted": len(rows), "successful": len(successful),
              "failed": len(rows) - len(successful)}
    for key in keys:
        values = [row["metrics"][key] for row in successful if row["metrics"][key] is not None]
        result[key] = {"mean": sum(values) / len(values) if values else None,
                       "queries": len(values)}
    return result
