"""Повторная генерация на неизменных кандидатах сохранённого прогона.

python -m evaluation.repeat_answers --baseline evaluation/results/<имя>.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from backend.prompts import SYSTEM_PROMPT
from backend.search import ANSWER_MODEL

from .metrics import average_metrics, score_answer
from .report import render_report
from .run import Cache, complete, digest, load_dataset, safe_error, save_json


def fixed_inputs(baseline, dataset, queries, query_ids, configuration, *, expected_prompt=SYSTEM_PROMPT):
    """Не смешиваем данные, промпты и разные наборы кандидатов."""
    if baseline["system_prompt"] != expected_prompt or baseline["answer_model"] != ANSWER_MODEL:
        raise ValueError("Модель или промпт отличаются от исходного прогона")
    if baseline["dataset_sha256"] != digest(dataset) or not baseline.get("index_unchanged"):
        raise ValueError("Другой датасет или исходный индекс менялся во время прогона")
    approved = {query["id"]: query for query in queries}
    if len(set(query_ids)) != len(query_ids) or not set(query_ids) <= approved.keys():
        raise ValueError("Нужны уникальные ID согласованных запросов")
    inputs = []
    for query_id in query_ids:
        rows = [row for row in baseline["rows"] if row["query_id"] == query_id
                and row["configuration"] == configuration and row["status"] == "success"]
        if len(rows) != 1 or rows[0]["query"] != approved[query_id]["query"]:
            raise ValueError("Нет однозначного исходного контекста: " + query_id)
        row = rows[0]
        # Проверяет, что все ID известны разметке и ответ относится к контексту.
        score_answer(row["answer"], approved[query_id],
                     [book["id"] for book in row["dense_candidates"]],
                     [book["id"] for book in row["llm_candidates"]])
        inputs.append((approved[query_id], row))
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--dataset", default="evaluation/book_search_v4.json")
    parser.add_argument("--configuration", default="20-to-5")
    parser.add_argument("--query-ids", nargs="+", default=["q008", "q029", "q005", "q012", "q039"])
    parser.add_argument("--samples", nargs="+", default=["stability-1", "stability-2", "stability-3"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--use-baseline-prompt", action="store_true",
                        help="Повторить историческую инструкцию вместо текущей")
    parser.add_argument("--cache-dir", default="evaluation/cache")
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()
    if len(set(args.samples)) != len(args.samples):
        parser.error("Повторяющиеся номера samples")
    load_dotenv(".env", encoding="utf-8-sig")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    dataset, queries = load_dataset(args.dataset, 10000)
    prompt = baseline["system_prompt"] if args.use_baseline_prompt else SYSTEM_PROMPT
    inputs = fixed_inputs(baseline, dataset, queries, args.query_ids, args.configuration,
                          expected_prompt=prompt)
    cache = Cache(args.cache_dir, offline=args.offline)
    report = {key: baseline[key] for key in (
        "dataset_sha256", "dataset_version", "index_sha256", "collection",
        "embedding_model", "rerank_model", "answer_model", "system_prompt",
    )}
    report.update({"started_at": datetime.now(timezone.utc).isoformat(),
                   "experiment": "fixed-context-answer-stability",
                   "baseline_path": str(args.baseline), "baseline_sha256": digest(baseline),
                   "dataset_path": args.dataset, "offline": args.offline,
                   "thinking": args.thinking,
                   "index_note": "Кандидаты из сохранённого снимка; живой Qdrant не используется.",
                   "generation_note": "Режим thinking указан отдельно; остальные параметры как в приложении. sample — ключ кеша, не seed модели.",
                   "rows": []})
    output = Path(args.output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    configs = [args.configuration + " / " + sample for sample in args.samples]
    for query, source in inputs:
        books = source["llm_candidates"]
        for sample, config in zip(args.samples, configs):
            row = {key: source[key] for key in (
                "query_id", "query", "dense_candidates", "rerank_candidates", "llm_candidates",
            )}
            row.update({"configuration": config, "sample": sample, "status": "error",
                        "context_sha256": digest(books)})
            try:
                result, timing = cache.get("answer", {
                    "model": ANSWER_MODEL, "prompt": prompt, "query": query["query"],
                    "books": books, "thinking": args.thinking, "format": "json_object", "sample": sample,
                }, lambda: complete(query["query"], books, thinking=args.thinking,
                                    system_prompt=prompt))
                row.update(result)
                row["metrics"] = score_answer(result["answer"], query,
                    [book["id"] for book in source["dense_candidates"]], [book["id"] for book in books])
                row.update({"answer_timing": timing, "status": "success"})
            except Exception as error:
                row["error"] = {"stage": "answer", "errors": safe_error(error)}
            report["rows"].append(row)
            save_json(output, report)
            print(query["id"], sample, row["status"], flush=True)
    report["new_calls_by_stage"] = cache.new_calls
    report["paired_query_ids"] = sorted(set.intersection(*[
        {row["query_id"] for row in report["rows"] if row["configuration"] == name
         and row["status"] == "success"} for name in configs]))
    report["summary"], report["paired_summary"] = {}, {}
    for name in configs:
        rows = [row for row in report["rows"] if row["configuration"] == name]
        summary = average_metrics(rows)
        usage = [row["usage"] for row in rows if row["status"] == "success" and row.get("usage")]
        for key in ("prompt_tokens", "completion_tokens"):
            summary["mean_" + key] = sum(item[key] for item in usage) / len(usage) if usage else None
        report["summary"][name] = summary
        report["paired_summary"][name] = average_metrics([
            row for row in rows if row["query_id"] in report["paired_query_ids"]])
    save_json(output, report)
    heading = ("# Устойчивость финальной генерации\n\n"
               f"Режим thinking: {args.thinking}.\n\n"
               "Повторы используют один промпт и одинаковые сохранённые контексты. "
               "Номера повторов не передаются модели как seed. "
               "Эмбеддинги, реранкер и живая база не вызываются.\n\n")
    output.with_suffix(".md").write_text(heading + render_report(report), encoding="utf-8")
    print("Report:", output.resolve())
    if any(row["status"] != "success" for row in report["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
