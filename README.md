# 📚 Book Expert

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

- **Приложение** доступно по `http://localhost:5000`

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
