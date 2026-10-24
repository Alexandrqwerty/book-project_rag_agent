"""Пересчитать опубликованные ответы без API, ключей и Qdrant."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from backend.search import render_answer

from .metrics import average_metrics, score_answer
from .report import render_report
from .run import digest, load_dataset, save_json


def rescore(report, dataset, queries):
    if report["dataset_sha256"] != digest(dataset):
        raise ValueError("Разметка отличается от использованной в эксперименте")
    result = copy.deepcopy(report)
    lookup = {query["id"]: query for query in queries}
    for row in result["rows"]:
        query = lookup[row["query_id"]]
        if query["query"] != row["query"]:
            raise ValueError("Текст запроса не соответствует разметке")
        if row["status"] != "success":
            continue
        render_answer(json.dumps(row["answer"], ensure_ascii=False), row["llm_candidates"])
        metrics = score_answer(row["answer"], query,
                               [book["id"] for book in row["dense_candidates"]],
                               [book["id"] for book in row["llm_candidates"]])
        # Отсутствующие в старых форматах поля допускаются, несовпадение — нет.
        if any(metrics[key] != value for key, value in row.get("metrics", {}).items()
               if key in metrics):
            raise ValueError("Сохранённые метрики расходятся с пересчётом")
        row["metrics"] = metrics
    configs = sorted({row["configuration"] for row in result["rows"]})
    common = set.intersection(*[
        {row["query_id"] for row in result["rows"]
         if row["configuration"] == config and row["status"] == "success"}
        for config in configs
    ])
    result["paired_query_ids"] = sorted(common)
    result["summary"], result["paired_summary"] = {}, {}
    for config in configs:
        rows = [row for row in result["rows"] if row["configuration"] == config]
        summary = average_metrics(rows)
        successful = [row for row in rows if row["status"] == "success"]
        usages = [row["usage"] for row in successful if row.get("usage")]
        summary["usage_measured_queries"] = len(usages)
        for key in ("prompt_tokens", "completion_tokens"):
            summary["mean_" + key] = mean(item[key] for item in usages) if usages else None
        times = [row["answer_timing"]["seconds"] for row in successful]
        summary["mean_answer_seconds"] = mean(times) if times else None
        summary["median_answer_seconds"] = median(times) if times else None
        durations = [row["original_stage_seconds_sum"] for row in successful
                     if "original_stage_seconds_sum" in row]
        summary["mean_original_stage_seconds_sum"] = mean(durations) if durations else None
        result["summary"][config] = summary
        result["paired_summary"][config] = average_metrics([
            row for row in rows if row["query_id"] in common])
    result["replay_note"] = "Пересчёт сохранённых ответов; новых вызовов моделей и базы нет."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/replay"))
    args = parser.parse_args()
    references = Path(__file__).parent / "reference"
    manifest = json.loads((references / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        path = references / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Повреждён сохранённый отчёт: " + str(path))
        dataset, queries = load_dataset(item["dataset"], 10000)
        result = rescore(json.loads(path.read_text(encoding="utf-8")), dataset, queries)
        output = args.output_dir / item["file"]
        save_json(output, result)
        output.with_suffix(".md").write_text(render_report(result), encoding="utf-8")
        print(item["file"], "OK", len(result["rows"]), "ответов/попыток")
    print("Отчёты:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
