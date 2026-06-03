# Бэкенд для агента Книжный эксперт

import os
import json
from typing import Annotated, TypedDict, List

import requests
from dotenv import load_dotenv
from flask import Flask, render_template_string, request, jsonify

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from qdrant_client import QdrantClient
from openai import OpenAI

# Local imports
from upload import add_books_from_json
from prompts import SYSTEM_PROMPT, HTML_TEMPLATE, TOOL_DESCRIPTION

load_dotenv(dotenv_path=".env", encoding="utf-8")

# Ключи и настройка конфигурации модели

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

# Функции поиска книг

def get_embedding(text: str):
    try:
        res = openai_client.embeddings.create(
            model="openai/text-embedding-3-small",
            input=[text],
        )
        return res.data[0].embedding
    except Exception as e:
        print(f"Ошибка эмбеддинга: {e}")
        return None

def search_books(query: str, limit: int = 20):
    query_vector = get_embedding(query)
    if not query_vector:
        return []
    results = local_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using="dense-vector",
        limit=limit,
        with_payload=True,
    )
    return results.points

def rerank(query: str, documents, top_n: int = 5):
    docs = []
    for doc in documents:
        text = (
            f"Название: {doc.payload.get('title', '')}. "
            f"Автор: {doc.payload.get('author', '')}. "
            f"Жанр: {doc.payload.get('category', '')}. "
            f"Описание: {doc.payload.get('enriched_description', doc.payload.get('description', ''))}"
        )
        docs.append(text)
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/rerank",
            headers={
                "Authorization": f"Bearer {openai_client.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "cohere/rerank-4-pro",
                "query": query,
                "documents": docs,
                "top_n": top_n,
            },
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Ошибка реранкера: {e}")
        return None

def search_with_rerank(query: str, limit: int = 5):
    candidates = search_books(query, limit=20)
    if not candidates:
        return []
    rerank_result = rerank(query, candidates, top_n=limit)
    if not rerank_result or "results" not in rerank_result:
        return candidates[:limit]
    reranked = []
    for item in rerank_result["results"]:
        idx = item["index"]
        score = item["relevance_score"]
        candidate = candidates[idx]
        candidate.score = score
        reranked.append(candidate)
    return reranked

# Создание агента

@tool
def search_books_with_rerank(query: str) -> str:
    """Ищет книги по описанию. Используй для поиска книг по темам, жанрам, сюжетам."""
    results = search_with_rerank(query, limit=5)
    if not results:
        return "Ничего не найдено по вашему запросу"
    output = []
    for i, hit in enumerate(results, 1):
        output.append(f"{i}. {hit.payload.get('title')} — {hit.payload.get('author')}")
        output.append(f"   Жанр: {hit.payload.get('category')}")
        output.append(f"   Год: {hit.payload.get('year')}")
        output.append("")
    return "\n".join(output)

tools = [search_books_with_rerank]
tool_node = ToolNode(tools=tools)

llm = ChatOpenAI(
    base_url="https://foundation-models.api.cloud.ru/v1",
    api_key=CLOUDRU_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0.3,
)

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[List, add_messages]

def agent(state: State):
    msgs = state.get("messages", [])
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

def route(state: State):
    msgs = state.get("messages", [])
    if not msgs:
        return END
    last = msgs[-1]
    if hasattr(last, "tool_calls") and getattr(last, "tool_calls"):
        return "tools"
    return END

graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge(START, "agent")
graph = graph_builder.compile()

# Flask

flask_app = Flask(__name__)

@flask_app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@flask_app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({'response': 'Пожалуйста, задайте вопрос'})
    try:
        final_state = graph.invoke({"messages": [HumanMessage(content=user_input)]})
        answer = final_state['messages'][-1].content
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'response': f'Ошибка: {str(e)}'})

@flask_app.route('/upload_books', methods=['POST'])
def upload_books():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Файл не выбран'}), 400
    temp_path = "temp_upload.json"
    file.save(temp_path)
    try:
        added, message = add_books_from_json(temp_path)
        return jsonify({'message': message, 'added': added})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=5000, debug=False)
