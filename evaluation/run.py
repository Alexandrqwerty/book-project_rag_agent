"""python -m evaluation.run --limit 5; повтор без внешних API: --offline."""

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from backend.catalog import book_id, source_fingerprint, validate_catalog
from backend.prompts import SYSTEM_PROMPT
from backend.search import (
    ANSWER_MODEL,
    DEFAULT_RERANK_MODEL,
    render_answer,
    request_answer,
    rerank_books,
    search_books_by_vector,
)
from backend.storage import COLLECTION_NAME, VECTOR_SIZE, validate_collection
from backend.upload import (
    DENSE_MODEL,
    DENSE_VERSION,
    ENRICHMENT_VERSION,
    get_dense_vector,
    get_llm_client,
    get_qdrant_client,
    valid_dense,
)

from .metrics import average_metrics, score_answer
from .report import render_report


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


class Cache:
    """Ошибки не кешируются. offline запрещает новые вызовы внешних моделей."""

    def __init__(self, root, offline=False):
        self.root, self.offline = Path(root), offline
        self.new_calls = {}

    def get(self, stage, parameters, produce):
        key = digest({"cache_version": 1, "stage": stage, "parameters": parameters})
        path = self.root / stage / (key + ".json")
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            if item["key"] != key or item["value_sha256"] != digest(item["value"]):
                raise ValueError("Повреждённый кеш: " + str(path))
            return item["value"], {"cached": True, "seconds": item["seconds"]}
        if self.offline:
            raise RuntimeError("Нет кеша этапа " + stage + "; внешний API запрещён")
        started = perf_counter()
        self.new_calls[stage] = self.new_calls.get(stage, 0) + 1
        value = produce()
        elapsed = perf_counter() - started
        save_json(path, {"key": key, "value": value, "value_sha256": digest(value),
                         "seconds": elapsed})
        return value, {"cached": False, "seconds": elapsed}


def load_dataset(path, limit):
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    catalog = dataset["catalog"]
    validate_catalog(catalog)
    ids = {book["book_id"] for book in catalog}
    if len(ids) != len(catalog) or any(book_id(book) != book["book_id"] for book in catalog):
        raise ValueError("Некорректные ID оценочного каталога")
    if digest(catalog) != dataset["catalog_sha256"]:
        raise ValueError("Хеш каталога не соответствует содержимому")
    evidence = dataset.get("database_evidence_snapshot")
    if evidence is not None:
        if digest(evidence) != dataset["database_evidence_sha256"]:
            raise ValueError("Хеш описаний базы не соответствует снимку")
        if len(evidence) != len(ids) or {book["book_id"] for book in evidence} != ids:
            raise ValueError("Снимок описаний не соответствует составу каталога")
    queries = [query for query in dataset["queries"] if query["include_in_metrics"]]
    if len({q["id"] for q in dataset["queries"]}) != len(dataset["queries"]):
        raise ValueError("Повторные ID запросов")
    for query in queries:
        judgments = query["judgments"]
        if query["review_status"] != "human_reviewed" or len(judgments) != len(ids):
            raise ValueError("Разметка не согласована: " + query["id"])
        if {item["book_id"] for item in judgments} != ids or any(
            type(item["grade"]) is not int or item["grade"] not in {0, 1, 2}
            for item in judgments
        ):
            raise ValueError("Неполная или некорректная разметка: " + query["id"])
        if query["expected_behavior"] not in {"recommend", "no_match"}:
            raise ValueError("Запрос пока нельзя оценивать: " + query["id"])
    if not queries:
        raise ValueError("Нет согласованных запросов")
    return dataset, queries[:limit]


def check_index(catalog, database_evidence_snapshot=None):
    """Полный индекс нужен до сравнения; разные версии данных не смешиваем."""
    client = get_qdrant_client()
    validate_collection(client)
    points, offset = [], None
    while True:
        page, offset = client.scroll(COLLECTION_NAME, limit=100, offset=offset,
                                     with_vectors=True, with_payload=True)
        points.extend(page)
        if offset is None:
            break
    expected = {book["book_id"]: book for book in catalog}
    evidence = {book["book_id"]: book for book in database_evidence_snapshot or []}
    if {str(point.id) for point in points} != expected.keys():
        raise ValueError("Индекс должен содержать ровно все книги оценочного каталога")
    snapshot = []
    for point in sorted(points, key=lambda item: str(item.id)):
        payload = point.payload or {}
        source = expected[str(point.id)]
        metadata = payload.get("indexing", {})
        if source_fingerprint(payload) != source_fingerprint(source):
            raise ValueError("Изменились исходные данные: " + source["title"])
        if evidence and any(payload.get(key) != value for key, value in evidence[str(point.id)].items()
                            if key != "book_id"):
            raise ValueError("Описание базы отличается от оценочного снимка: " + source["title"])
        if (metadata.get("dense_version") != DENSE_VERSION
                or metadata.get("enrichment_version") != ENRICHMENT_VERSION):
            raise ValueError("Устаревшая версия индекса: " + source["title"])
        dense = (point.vector or {}).get("dense-vector")
        if not valid_dense(dense):
            raise ValueError("Некорректный вектор: " + source["title"])
        snapshot.append({"id": str(point.id), "payload": payload, "dense": dense})
    return digest(snapshot)


def complete(query, books, *, thinking="disabled", system_prompt=SYSTEM_PROMPT):
    if system_prompt == SYSTEM_PROMPT:
        response = request_answer(query, books, thinking=thinking)
    else:
        # Историческая инструкция из опубликованного отчёта; формат запроса тот же.
        context = [{key: book[key] for key in (
            "id", "title", "author", "category", "year", "description", "enriched_description",
        )} for book in books]
        response = get_llm_client().chat.completions.create(
            model=ANSWER_MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": json.dumps(
                          {"query": query, "books": context}, ensure_ascii=False)}],
            extra_body={"thinking": {"type": thinking}},
            response_format={"type": "json_object"},
        )
    content = response.choices[0].message.content
    rendered = render_answer(content, books)
    usage = response.usage.model_dump(mode="json") if response.usage else None
    return {"answer": json.loads(content), "rendered": rendered, "usage": usage,
            "response_model": response.model,
            "reasoning_present": bool(getattr(response.choices[0].message, "reasoning_content", None)),
            "system_fingerprint": getattr(response, "system_fingerprint", None)}


def safe_error(error):
    # Сообщение провайдера может содержать секреты; сохраняем только тип и статус.
    chain, current, seen = [], error, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        chain.append({"type": type(current).__name__, "http_status": status})
        current = current.__cause__
    return chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evaluation/book_search_v4.json")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--port", type=int, default=6335,
                        help="Порт отдельного локального Qdrant для экспериментов")
    parser.add_argument("--contexts", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--all-catalog", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir", default="evaluation/cache")
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--sample", default="1", help="Новый номер повторения заново вызывает LLM")
    parser.add_argument("--prompt-reference", type=Path,
                        help="JSON исторического отчёта: использовать его system_prompt")
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    args = parser.parse_args()
    if args.limit < 1 or not args.contexts or not all(1 <= n <= args.candidates for n in args.contexts):
        parser.error("Нужны limit >= 1 и 1 <= contexts <= candidates")
    for price in (args.input_usd_per_million, args.output_usd_per_million):
        if price is not None and (not math.isfinite(price) or price < 0):
            parser.error("Цена должна быть конечной и неотрицательной")
    load_dotenv(".env", encoding="utf-8-sig")
    if not 1 <= args.port <= 65535 or args.port == 6333:
        parser.error("Используйте отдельный Qdrant, например порт 6335, а не рабочий 6333")
    os.environ["QDRANT_HOST"] = "127.0.0.1"
    os.environ["QDRANT_PORT"] = str(args.port)
    get_qdrant_client.cache_clear()
    dataset, queries = load_dataset(args.dataset, args.limit)
    prompt = (json.loads(args.prompt_reference.read_text(encoding="utf-8"))["system_prompt"]
              if args.prompt_reference else SYSTEM_PROMPT)
    if not isinstance(prompt, str) or not prompt.strip():
        parser.error("Пустой или некорректный system_prompt")
    index_hash = check_index(dataset["catalog"], dataset.get("database_evidence_snapshot"))
    cache = Cache(args.cache_dir, args.offline)
    rerank_model = os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL)
    configs = {f"{args.candidates}-to-{n}": n for n in sorted(set(args.contexts))}
    if args.all_catalog:
        configs["all-catalog"] = len(dataset["catalog"])
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": digest(dataset), "index_sha256": index_hash,
        "dataset_version": dataset["version"], "dataset_path": str(Path(args.dataset)),
        "collection": COLLECTION_NAME, "embedding_model": DENSE_MODEL,
        "rerank_model": rerank_model, "answer_model": ANSWER_MODEL,
        "system_prompt": prompt, "sample": args.sample,
        "offline": args.offline, "rows": [],
        "cost_note": "LLM estimate uses supplied flat input/output prices; cache discounts and failed attempts are not included. Embedding/rerank billing is not measured.",
    }
    output = Path(args.output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    for query in queries:
        dense, ranked, shared, shared_error = [], [], {}, None
        stage = "embedding"
        try:
            vector, shared[stage] = cache.get(stage, {
                "model": DENSE_MODEL, "dimensions": VECTOR_SIZE,
                "input_type": "query", "text": query["query"], "version": DENSE_VERSION,
            }, lambda: get_dense_vector(query["query"], input_type="query"))
            if not valid_dense(vector):
                raise ValueError("Некорректный кешированный вектор")
            stage = "dense"
            dense, shared[stage] = cache.get(stage, {
                "index": index_hash, "vector": digest(vector), "limit": args.candidates,
                "code_version": 1,
            }, lambda: search_books_by_vector(vector, limit=args.candidates))
            stage = "rerank"
            ranked, shared[stage] = cache.get(stage, {
                "model": rerank_model, "query": query["query"], "books": dense,
                "top_n": max(args.contexts), "code_version": 1,
            }, lambda: rerank_books(query["query"], dense, limit=max(args.contexts)))
        except Exception as error:
            shared_error = {"stage": stage, "errors": safe_error(error)}
        for name, size in configs.items():
            row = {"query_id": query["id"], "query": query["query"], "configuration": name,
                   "status": "error", "shared_stages": shared.copy()}
            try:
                if name != "all-catalog" and shared_error:
                    row["error"] = shared_error
                    report["rows"].append(row)
                    print(query["id"], name, "ERROR", shared_error["stage"], flush=True)
                    continue
                if name == "all-catalog":
                    # Те же обогащения, что в индексе; порядок фиксирован по UUID.
                    points = get_qdrant_client().retrieve(
                        COLLECTION_NAME, ids=sorted(book["book_id"] for book in dataset["catalog"]),
                        with_payload=True, with_vectors=False,
                    )
                    context = [{"id": str(point.id), **point.payload} for point in sorted(points, key=lambda p: str(p.id))]
                else:
                    context = ranked[:size]
                row["dense_candidates"], row["llm_candidates"] = dense, context
                row["rerank_candidates"] = ranked if name != "all-catalog" else []
                stage = "answer"
                result, timing = cache.get(stage, {
                    "model": ANSWER_MODEL, "prompt": prompt, "query": query["query"],
                    "books": context, "thinking": "disabled", "format": "json_object",
                    "sample": args.sample,
                }, lambda: complete(query["query"], context, system_prompt=prompt))
                # Валидируем повторно и при чтении из кеша.
                render_answer(json.dumps(result["answer"], ensure_ascii=False), context)
                row.update(result)
                row["answer_timing"] = timing
                row["metrics"] = score_answer(result["answer"], query,
                                               [book["id"] for book in dense],
                                               [book["id"] for book in context])
                stages = ([] if name == "all-catalog" else list(shared.values())) + [timing]
                row["original_stage_seconds_sum"] = sum(item["seconds"] for item in stages)
                row["component_cache_misses"] = sum(not item["cached"] for key, item in shared.items()
                                                 if key in {"embedding", "rerank"}) if name != "all-catalog" else 0
                row["component_cache_misses"] += int(not timing["cached"])
                row["llm_cost_estimate_usd"] = None
                usage = result["usage"]
                if usage and args.input_usd_per_million is not None and args.output_usd_per_million is not None:
                    row["llm_cost_estimate_usd"] = (
                        usage["prompt_tokens"] * args.input_usd_per_million
                        + usage["completion_tokens"] * args.output_usd_per_million
                    ) / 1_000_000
                row["status"] = "success"
                if name == "all-catalog":
                    for key in ("dense_recall_strong", "dense_recall_including_partial"):
                        row["metrics"][key] = None
                print(query["id"], name, "OK", "precision", row["metrics"]["precision_strong"],
                      "coverage", row["metrics"]["coverage_capped_at_5"], flush=True)
            except Exception as error:
                row["error"] = {"stage": stage, "errors": safe_error(error)}
                print(query["id"], name, "ERROR", stage, flush=True)
            report["rows"].append(row)
            save_json(output, report)
        save_json(output, report)
        if shared_error and any(item["http_status"] == 429 for item in shared_error["errors"]):
            report["stopped_reason"] = "HTTP 429: remaining queries were not attempted"
            break
    report["index_unchanged"] = check_index(dataset["catalog"], dataset.get("database_evidence_snapshot")) == index_hash
    report["new_calls_by_stage"] = cache.new_calls
    report["summary"] = {name: average_metrics([row for row in report["rows"]
                                               if row["configuration"] == name]) for name in configs}
    common = set.intersection(*[
        {row["query_id"] for row in report["rows"] if row["configuration"] == name and row["status"] == "success"}
        for name in configs
    ])
    report["paired_query_ids"] = sorted(common)
    report["paired_summary"] = {name: average_metrics([row for row in report["rows"]
        if row["configuration"] == name and row["query_id"] in common]) for name in configs}
    for name, summary in report["summary"].items():
        rows = [row for row in report["rows"] if row["configuration"] == name and row["status"] == "success"]
        tokens = [row["usage"] for row in rows if row["usage"] is not None]
        summary["usage_measured_queries"] = len(tokens)
        for key in ("prompt_tokens", "completion_tokens"):
            summary["mean_" + key] = sum(usage[key] for usage in tokens) / len(tokens) if tokens else None
        summary["mean_original_stage_seconds_sum"] = (
            sum(row["original_stage_seconds_sum"] for row in rows) / len(rows) if rows else None
        )
    save_json(output, report)
    output.with_suffix(".md").write_text(render_report(report), encoding="utf-8")
    print("Report:", output.resolve())
    if not report["index_unchanged"]:
        raise SystemExit("Индекс изменился во время прогона; сравнение недействительно")
    if any(row["status"] != "success" for row in report["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
