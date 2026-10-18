"""Сборка графа поиска и запуск веб-приложения."""

import logging
import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from .search import (
    generate_answer,
    rerank_enabled,
    retrieve_books,
    validate_search_config,
)
from .storage import prepare_collection
from .upload import add_books_from_json, get_qdrant_client
from .web import create_app

load_dotenv(dotenv_path=".env", encoding="utf-8-sig")


class State(TypedDict):
    query: str
    books: list[dict]
    answer: str


def retrieve(state: State):
    books = retrieve_books(state["query"])
    return {
        "books": books,
        "answer": "" if books else "В каталоге не найдено книг для ответа. Загрузите каталог или проверьте индекс.",
    }


def generate(state: State):
    return {"answer": generate_answer(state["query"], state["books"])}


def route_after_search(state: State):
    return "generate" if state["books"] else END


def build_graph():
    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges("retrieve", route_after_search, {"generate": "generate", END: END})
    builder.add_edge("generate", END)
    return builder.compile()


graph = build_graph()


def answer_question(user_input: str) -> str:
    return graph.invoke({"query": user_input, "books": [], "answer": ""})["answer"]


flask_app = create_app(answer_question, add_books_from_json)


def main():
    logging.basicConfig(level=logging.INFO)
    for key_name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        if not os.getenv(key_name, "").strip():
            raise ValueError(f"{key_name} не найден. Заполните .env по примеру .env.example.")
    validate_search_config()
    logging.info("Режим поиска: %s", "dense + rerank" if rerank_enabled() else "dense")
    prepare_collection(get_qdrant_client())
    flask_app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
