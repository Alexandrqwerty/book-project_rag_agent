# 📚 Book Project

Приложение с раг агентом для загрузки и поска книг. Всё собирается и запускается в Docker.

## Запуск
```bash
# Клонировать репозиторий и перейти в каталог
git clone https://github.com/Alexandrqwerty/book-project_rag_agent.git
cd book-project_rag_agent

# При необходимости скопировать .env
cp .env.example .env

# Собрать и запустить контейнеры
docker compose up --build
```

- **Frontend** доступен по `http://localhost:5000`
- **Backend** доступен по `http://localhost:8000` (Swagger UI – `http://localhost:8000/docs`)

## Структура проекта
```
/home/alex/book-project_new/
├─ .env.example
├─ docker-compose.yml
├─ Dockerfile.backend
├─ Dockerfile.frontend
├─ requirements.txt
├─ backend/
│   ├─ app.py
│   ├─ prompts.py
│   └─ upload.py
└─ frontend/
    ├─ app.py
    └─ prompts.py
```

## Конфигурация
Переменные окружения берутся из `.env` (или из системы). Основные:
- `UPLOAD_DIR` – каталог для загруженных файлов (по умолчанию `./uploads`).
- `HOST` – хост бэкенда (по умолчанию `0.0.0.0`).
- `PORT` – порт бэкенда (по умолчанию `8000`).

## Лицензия
MIT – см. файл `LICENSE`.
