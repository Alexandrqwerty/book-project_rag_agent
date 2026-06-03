# -*- coding: utf-8 -*-
# Промпты и фронтэнд визуал

SYSTEM_PROMPT = """Вы книжный эксперт. Отвечайте на русском, рекомендуйте книги ТОЛЬКО из результатов поиска. Правило: НЕ используйте символы | (вертикальная черта) в ответах. И без спойлеров. Выведи список из полученных наиболее подходящих книг с краткой информацией и ответ на естественном языке"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Книжный эксперт</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 20px;
        }
        .chat-container {
            background: white;
            border-radius: 24px;
            width: 100%;
            max-width: 700px;
            height: 80vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 { margin: 0; font-size: 24px; }
        .chat-header p { margin: 5px 0 0; opacity: 0.9; font-size: 14px; }
        .upload-btn {
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.5);
            padding: 6px 15px;
            font-size: 12px;
            margin-top: 10px;
            cursor: pointer;
            border-radius: 20px;
            display: inline-block;
            transition: all 0.3s;
        }
        .upload-btn:hover {
            background: rgba(255,255,255,0.3);
            transform: scale(1.02);
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
        }
        .user { justify-content: flex-end; }
        .bot { justify-content: flex-start; }
        .message-content {
            max-width: 70%;
            padding: 10px 15px;
            border-radius: 18px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        .user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .bot .message-content {
            background: #e9ecef;
            color: #333;
        }
        .input-area {
            display: flex;
            padding: 15px;
            background: white;
            border-top: 1px solid #dee2e6;
            gap: 10px;
        }
        textarea {
            flex: 1;
            padding: 10px;
            border: 1px solid #dee2e6;
            border-radius: 20px;
            resize: none;
            font-family: inherit;
            font-size: 14px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { opacity: 0.9; }
        .file-input { display: none; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>📚 Книжный эксперт</h1>
            <p>Спросите о любой книге</p>
            <input type="file" id="fileInput" class="file-input" accept=".json">
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                📁 Загрузить книги (JSON)
            </button>
        </div>
        <div class="messages" id="messages">
            <div class="bot">
                <div class="message-content">👋 Привет! Спроси меня о книгах. Например: "что почитать про космос?"</div>
            </div>
        </div>
        <div class="input-area">
            <textarea id="input" placeholder="Напишите ваш вопрос..." rows="2"></textarea>
            <button onclick="sendMessage()">Отправить</button>
        </div>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('input');
            const msg = input.value.trim();
            if (!msg) return;

            const messagesDiv = document.getElementById('messages');
            messagesDiv.innerHTML += '<div class="message user"><div class="message-content">' + escapeHtml(msg) + '</div></div>';
            input.value = '';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            const loadingId = 'loading-' + Date.now();
            messagesDiv.innerHTML += '<div class="bot" id="' + loadingId + '"><div class="message-content">🤔 Думаю...</div></div>';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg})
                });
                const data = await response.json();
                document.getElementById(loadingId).remove();
                messagesDiv.innerHTML += '<div class="bot"><div class="message-content">' + escapeHtml(data.response) + '</div></div>';
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            } catch (err) {
                document.getElementById(loadingId).remove();
                messagesDiv.innerHTML += '<div class="bot"><div class="message-content">❌ Ошибка: ' + escapeHtml(err.message) + '</div></div>';
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        }

        async function uploadBook() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            if (!file) return;

            const messagesDiv = document.getElementById('messages');
            messagesDiv.innerHTML += '<div class="bot"><div class="message-content">⏳ Загрузка и индексация книг...</div></div>';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload_books', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                messagesDiv.innerHTML += '<div class="bot"><div class="message-content">✅ ' + escapeHtml(data.message || 'Книги успешно загружены!') + '</div></div>';
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            } catch (err) {
                messagesDiv.innerHTML += '<div class="bot"><div class="message-content">❌ Ошибка загрузки: ' + escapeHtml(err.message) + '</div></div>';
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            fileInput.value = '';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        document.getElementById('input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        document.getElementById('fileInput').addEventListener('change', function() {
            if (this.files && this.files[0]) {
                uploadBook();
            }
        });
    </script>
</body>
</html>
"""

# Описание инструмента search_books_with_rerank (для bind_tools)
TOOL_DESCRIPTION = "Ищет книги по описанию. Используй для поиска книг по темам, жанрам, сюжетам."
