const messages = document.getElementById("messages");
const conversation = document.getElementById("conversation");
const welcome = document.getElementById("welcome");
const input = document.getElementById("input");
const sendButton = document.getElementById("sendButton");
const sendLabel = document.getElementById("sendLabel");
const uploadButton = document.getElementById("uploadButton");
const uploadLabel = document.getElementById("uploadLabel");
const catalogStatus = document.getElementById("catalogStatus");
const fileInput = document.getElementById("fileInput");
const searchForm = document.getElementById("searchForm");
let searching = false;
let uploading = false;

function makeIcon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#icon-${name}`);
    svg.append(use);
    return svg;
}

function scrollToLatest() {
    conversation.scrollTop = conversation.scrollHeight;
}

function resizeInput() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
    sendButton.disabled = searching || !input.value.trim();
}

function updateViewport() {
    // На телефоне учитываем видимую высоту при открытой экранной клавиатуре.
    if (window.visualViewport) {
        document.documentElement.style.setProperty("--app-height", `${window.visualViewport.height}px`);
    }
}

function addMessage(text, role = "bot", label = "Книжный эксперт") {
    welcome.hidden = true;
    const message = document.createElement("div");
    message.className = `message ${role}`;
    if (role === "bot") {
        const meta = document.createElement("div");
        meta.className = "message-meta";
        const avatar = document.createElement("span");
        avatar.className = "message-avatar";
        avatar.append(makeIcon("book"));
        meta.append(avatar, document.createTextNode(label));
        message.append(meta);
    }
    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = text;
    message.append(content);
    messages.append(message);
    scrollToLatest();
    return content;
}

function showPending(content, text) {
    content.className = "message-content is-pending";
    const dots = document.createElement("span");
    dots.className = "waiting-dots";
    dots.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 3; i++) dots.append(document.createElement("i"));
    content.replaceChildren(dots, document.createTextNode(text));
}

function showStatus(content, text, state) {
    content.className = `message-content is-${state}`;
    content.textContent = text;
    scrollToLatest();
}

function renderAnswer(content, text) {
    // Все данные модели выводятся через textContent, а не как HTML.
    content.className = "message-content";
    content.replaceChildren();
    for (const block of text.trim().split(/\n\s*\n/)) {
        const [heading, ...description] = block.split("\n");
        const match = heading.match(/^(\d+)\.\s+(.+)$/);
        if (!match) {
            const paragraph = document.createElement("p");
            paragraph.className = "answer-summary";
            paragraph.textContent = block;
            content.append(paragraph);
            continue;
        }
        const card = document.createElement("article");
        card.className = "result-card";
        const number = document.createElement("span");
        number.className = "result-number";
        number.setAttribute("aria-hidden", "true");
        number.textContent = match[1].padStart(2, "0");
        const body = document.createElement("div");
        const title = document.createElement("h2");
        title.className = "result-title";
        title.textContent = match[2];
        body.append(title);
        let reason = description.join("\n");
        const label = reason.match(/^(Возможно, подойдёт|Подходит):\s*/);
        if (label) {
            const badge = document.createElement("span");
            badge.className = `match-tag${label[1].startsWith("Возможно") ? " partial" : ""}`;
            badge.textContent = label[1];
            body.append(badge);
            reason = reason.slice(label[0].length);
        }
        const detail = document.createElement("p");
        detail.className = "result-description";
        detail.textContent = reason;
        body.append(detail);
        card.append(number, body);
        content.append(card);
    }
    scrollToLatest();
}

async function readResponse(response) {
    let data;
    try {
        data = await response.json();
    } catch {
        throw new Error("Сервер вернул некорректный ответ.");
    }
    if (!response.ok) throw new Error(data.error || "Не удалось выполнить запрос.");
    return data;
}

async function sendMessage(event) {
    event.preventDefault();
    const message = input.value.trim();
    if (!message || searching) return;
    searching = true;
    sendButton.classList.add("is-loading");
    sendButton.setAttribute("aria-busy", "true");
    sendLabel.textContent = "Ищем…";
    addMessage(message, "user");
    input.value = "";
    resizeInput();
    const status = addMessage("");
    showPending(status, "Подбираем книги по вашему запросу…");
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });
        const data = await readResponse(response);
        if (typeof data.response !== "string" || !data.response.trim()) {
            throw new Error("Сервер вернул пустой ответ. Попробуйте ещё раз.");
        }
        renderAnswer(status, data.response);
    } catch (error) {
        showStatus(status, error.message, "error");
    } finally {
        searching = false;
        sendButton.classList.remove("is-loading");
        sendButton.removeAttribute("aria-busy");
        sendLabel.textContent = "Найти книгу";
        resizeInput();
        scrollToLatest();
        // Не открываем экранную клавиатуру повторно после ответа на телефоне.
        if (window.matchMedia("(hover: hover) and (pointer: fine)").matches) input.focus();
    }
}

async function uploadBooks() {
    const file = fileInput.files[0];
    if (!file || uploading) return;
    uploading = true;
    uploadButton.disabled = true;
    uploadButton.classList.add("is-loading");
    uploadButton.setAttribute("aria-busy", "true");
    uploadLabel.textContent = "Загрузка…";
    const status = addMessage("", "bot", "Каталог книг");
    showPending(status, `Загружаем и обрабатываем «${file.name}»…`);
    const formData = new FormData();
    formData.append("file", file);
    try {
        const response = await fetch("/upload_books", { method: "POST", body: formData });
        const data = await readResponse(response);
        const state = data.status === "success" ? "success" : data.status === "partial" ? "warning" : "error";
        const errors = (data.errors || []).map((item) => `${item.title} — ${item.author}: ${item.error}`);
        showStatus(status, [data.message, ...errors].join("\n"), state);
        if (data.status === "success" || data.status === "partial") {
            catalogStatus.textContent = `Последняя загрузка: ${file.name}`;
        }
    } catch (error) {
        showStatus(status, error.message, "error");
    } finally {
        uploading = false;
        uploadButton.disabled = false;
        uploadButton.classList.remove("is-loading");
        uploadButton.removeAttribute("aria-busy");
        uploadLabel.textContent = "Загрузить каталог";
        fileInput.value = "";
        scrollToLatest();
    }
}

searchForm.addEventListener("submit", sendMessage);
input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        searchForm.requestSubmit();
    }
});
document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
        input.value = button.dataset.query;
        resizeInput();
        input.focus();
    });
});
uploadButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", uploadBooks);
window.visualViewport?.addEventListener("resize", updateViewport);
updateViewport();
resizeInput();
