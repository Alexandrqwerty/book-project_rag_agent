"""HTTP-маршруты и страница приложения."""

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .catalog import CatalogValidationError
from .search import SearchError


def create_app(
    answer_question: Callable[[str], str],
    import_books: Callable[[str], dict],
) -> Flask:
    """Связываем HTTP с поиском и загрузкой через переданные функции."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/chat")
    def chat():
        data = request.get_json(silent=True)
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, str) or not message.strip():
            return jsonify(error="Передайте непустой текст в поле message"), 400
        try:
            return jsonify(response=answer_question(message.strip()))
        except SearchError as error:
            app.logger.exception("Ошибка сервиса поиска")
            return jsonify(error=str(error)), 503
        except Exception:
            app.logger.exception("Ошибка поиска книг")
            return jsonify(error="Не удалось выполнить поиск. Попробуйте позже."), 500

    @app.post("/upload_books")
    def upload_books():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify(error="Файл не выбран"), 400

        temp_path = None
        try:
            # У каждого запроса свой файл; закрываем его перед чтением на Windows.
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                file.save(temp_file)
            report = import_books(str(temp_path))
            # Завершённая обработка возвращает отчёт даже при ошибках отдельных книг.
            return jsonify(report)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return jsonify(error="Файл должен содержать корректный JSON в UTF-8"), 400
        except CatalogValidationError as error:
            return jsonify(error=str(error)), 400
        except Exception:
            app.logger.exception("Ошибка загрузки каталога")
            return jsonify(error="Не удалось загрузить каталог. Попробуйте позже."), 500
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    return app
