# Фронтэнд для агента Книжный эесперт

from flask import Flask, render_template_string

from prompts import HTML_TEMPLATE

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    # Expose on port 8080 (or any port you configure in Docker)
    app.run(host='0.0.0.0', port=8080, debug=False)
