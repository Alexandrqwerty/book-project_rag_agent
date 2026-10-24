"""Читаемый отчёт из сохранённого JSON, без Qdrant и внешних API.

python -m evaluation.report evaluation/results/<имя>.json
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def number(value, digits=6):
    return f"{value:.{digits}f}" if value is not None else "—"


def query_details(report):
    groups = OrderedDict()
    for row in report["rows"]:
        groups.setdefault(row["query_id"], []).append(row)
    lines = ["## Подробно по запросам", "",
             "Dense score — косинусная близость в Qdrant. Rerank score — оценка NVIDIA-реранкера.",
             "Это разные шкалы: их нельзя сравнивать между собой или читать как вероятность правильного ответа.",
             "Оценки ниже округлены до шести знаков; исходная точность сохранена в JSON.",
             "Реранкер вернул только верхнюю часть списка. Для остальных dense-кандидатов его оценки не сохранены.",
             "Варианты с разным числом книг на входе LLM используют одну общую ранжировку.", ""]
    for query_id, rows in groups.items():
        lines += [f"### {query_id}: {rows[0]['query']}", "", "#### 1. После векторного поиска", ""]
        dense = next((row["dense_candidates"] for row in rows if row.get("dense_candidates")), [])
        if dense:
            lines += [f"Найдено кандидатов: {len(dense)}. Порядок соответствует выдаче Qdrant.", "",
                      "| Место dense | Книга | Автор | Год | Категория | Dense score |",
                      "| --- | --- | --- | --- | --- | ---: |"]
            for rank, book in enumerate(dense, 1):
                lines.append(f"| {rank} | {cell(book['title'])} | {cell(book['author'])} | "
                             f"{book['year']} | {cell(book['category'])} | {number(book.get('dense_score'))} |")
        else:
            lines += ["Список dense-кандидатов не сохранён; см. ошибки этапов ниже."]
        lines += ["", "#### 2. После реранкера", ""]
        reranked_rows = [row for row in rows if row["configuration"] != "all-catalog"
                         and row.get("llm_candidates")]
        # В старых отчётах весь top-10 находится в строке конфигурации 20-to-10.
        # Не делаем новый запрос и не придумываем оценки исключённым кандидатам.
        ranked = next((row["rerank_candidates"] for row in rows if row.get("rerank_candidates")), None)
        if ranked is None:
            ranked = max((row["llm_candidates"] for row in reranked_rows), key=len, default=[])
        dense_ranks = {book["id"]: rank for rank, book in enumerate(dense, 1)}
        if ranked:
            lines += [f"Сохранено результатов реранкера: {len(ranked)}.", ""]
            headers = ["Место rerank", "Место dense", "Книга", "Автор", "Год", "Dense score", "Rerank score"]
            headers += [f"Передана LLM ({row['configuration']})" for row in reranked_rows]
            lines += ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            for rank, book in enumerate(ranked, 1):
                values = [rank, dense_ranks.get(book["id"], "—"), book["title"], book["author"],
                          book["year"], number(book.get("dense_score")), number(book.get("rerank_score"))]
                values += ["да" if book["id"] in {item["id"] for item in row["llm_candidates"]}
                           else "нет" for row in reranked_rows]
                lines.append("| " + " | ".join(cell(value) for value in values) + " |")
            excluded = [book["title"] for book in dense if book["id"] not in {item["id"] for item in ranked}]
            if excluded:
                lines += ["", "За пределами сохранённого списка реранкера: " + "; ".join(excluded) + ".",
                          "Их rerank score неизвестны. Отсутствие в списке не означает оценку 0."]
        else:
            lines += ["Результаты реранкера не сохранены или этот вариант не использует реранкер."]
        lines += ["", "#### 3. Выбор LLM и финальный ответ", ""]
        for row in rows:
            lines += [f"**{row['configuration']}**", ""]
            if row["status"] != "success":
                lines += ["Ошибка этапа: " + row["error"]["stage"] + ". Финальный ответ не получен.", ""]
                continue
            context = row["llm_candidates"]
            names = {book["id"]: f"{book['title']} — {book['author']}" for book in context}
            selected = [names[item["id"]] for item in row["answer"]["recommendations"]]
            lines += [f"На входе LLM: {len(context)} книг. Максимум рекомендаций: 5.", "",
                      "Выбрала: " + ("; ".join(selected) if selected else "ни одной книги") + ".", ""]
            usage = row.get("usage")
            if usage:
                lines += [f"Токены LLM: вход {usage['prompt_tokens']}, выход {usage['completion_tokens']}.", ""]
            lines += ["Финальный ответ (сохранён без редактирования):", ""]
            # Отделяем исходный текст модели от пояснений отчёта.
            lines += ["> " + line if line else ">" for line in row["rendered"].splitlines()]
            lines += [""]
    return "\n".join(lines) + "\n"


def render_report(report):
    lines = ["# Сравнение конфигураций", "",
             "Версия разметки: " + str(report.get("dataset_version", "см. dataset_sha256 в JSON")), "",
             "Общие успешные запросы: " + str(len(report["paired_query_ids"])), "",
             "| Вариант | Успех / попытки | Строгая точность (2) | Поддержанные рекомендации (1+2) | Полнота с лимитом 5 | Входные токены LLM | Выходные токены LLM |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, summary in report["summary"].items():
        paired = report["paired_summary"][name]
        lines.append(f"| {name} | {summary['successful']}/{summary['attempted']} | "
                     f"{number(paired['precision_strong']['mean'], 3)} | "
                     f"{number(paired.get('precision_supported', {}).get('mean'), 3)} | "
                     f"{number(paired['coverage_capped_at_5']['mean'], 3)} | "
                     f"{number(summary['mean_prompt_tokens'], 1)} | {number(summary['mean_completion_tokens'], 1)} |")
    lines += ["", "Точность и полнота относятся только к запросам с подходящими книгами (оценка 2).",
              "Частичные совпадения (оценка 1) учитываются отдельно и не считаются полным соответствием.", "",
              "Показатель (1+2) оценивает поддержку выбранных книг данными. Он не проверяет, обозначена ли частичность в тексте; это требует ручного разбора.", "",
              "| Вариант | Правильные отказы при отсутствии ответа | Ошибочные отказы на разрешимых запросах | Среднее число частичных рекомендаций | Среднее число рекомендаций с оценкой 0 |",
              "| --- | --- | --- | --- | --- |"]
    for name, paired in report["paired_summary"].items():
        def fraction(key):
            item = paired[key]
            return f"{round(item['mean'] * item['queries'])}/{item['queries']}" if item['queries'] else "— (нет запросов)"
        lines.append(f"| {name} | {fraction('correct_no_match')} | {fraction('false_refusal')} | "
                     f"{number(paired['partial_count']['mean'], 3)} | {number(paired['irrelevant_count']['mean'], 3)} |")
    lines += ["", "Качество сравнивается только на общих успешных запросах. Ошибки API учитываются отдельно.",
              "Токены — средние по успешным ответам. Объяснения LLM автоматически не оценены.",
              "Модели: " + report["embedding_model"] + " → " + report["rerank_model"] + " → " + report["answer_model"] + ".",
              "", query_details(report)]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    path = args.report.with_suffix(".md")
    path.write_text(render_report(report), encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
