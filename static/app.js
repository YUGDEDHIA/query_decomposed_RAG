const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const input = document.getElementById("input");
const sendButton = document.getElementById("send");

// Identifies this conversation to the server (src/server.py's _sessions) so
// follow-up questions resolve against prior turns. Created once per page
// load, not persisted -- a reload starts a fresh conversation, matching the
// fact that the chat pane above isn't persisted across reload either.
const sessionId = crypto.randomUUID();

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "message user";
  el.innerHTML = `<div class="bubble"></div>`;
  el.querySelector(".bubble").textContent = text;
  chat.appendChild(el);
  scrollToBottom();
}

function addAssistantMessage() {
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = `
    <div class="bubble progress">Thinking...</div>
  `;
  chat.appendChild(el);
  scrollToBottom();
  return el;
}

function setProgress(el, message) {
  const bubble = el.querySelector(".bubble");
  bubble.className = "bubble progress";
  bubble.textContent = message;
  scrollToBottom();
}

function setAnswer(el, data) {
  const bubble = el.querySelector(".bubble");
  bubble.className = "bubble";
  bubble.innerHTML = marked.parse(data.answer);

  const meta = document.createElement("div");
  meta.className = "meta";

  const badge = document.createElement("span");
  if (data.intent === "casual") {
    badge.className = "grounded-badge neutral";
    badge.textContent = "Direct response";
  } else if (data.intent === "gmp_query") {
    badge.className = "grounded-badge neutral";
    badge.textContent = "Live GMP data (unofficial)";
  } else if (data.intent === "cross_company_query") {
    badge.className = "grounded-badge " + (data.fully_grounded ? "full" : "partial");
    badge.textContent = data.fully_grounded ? "Compared across indexed companies" : "Comparison unavailable";
  } else {
    badge.className = "grounded-badge " + (data.fully_grounded ? "full" : "partial");
    badge.textContent = data.fully_grounded ? "Fully grounded" : "Partially grounded";
  }
  meta.appendChild(badge);

  if (data.sources && data.sources.length) {
    const toggle = document.createElement("button");
    toggle.className = "sources-toggle";
    toggle.type = "button";
    toggle.textContent = `${data.sources.length} source(s)`;

    const list = document.createElement("div");
    list.className = "sources-list";
    list.innerHTML = data.sources
      .map(s => `<div>${s.company}${s.section ? " / " + s.section : ""} (page ${s.page_range[0]})</div>`)
      .join("");

    toggle.addEventListener("click", () => list.classList.toggle("open"));

    meta.appendChild(toggle);
    el.appendChild(meta);
    el.appendChild(list);
  } else {
    el.appendChild(meta);
  }

  scrollToBottom();
}

function setError(el, message) {
  const bubble = el.querySelector(".bubble");
  bubble.className = "bubble error";
  bubble.textContent = "Something went wrong: " + message;
  scrollToBottom();
}

function setBusy(busy) {
  input.disabled = busy;
  sendButton.disabled = busy;
}

function ask(question) {
  addUserMessage(question);
  const assistantEl = addAssistantMessage();
  setBusy(true);

  const source = new EventSource(
    "/api/chat?q=" + encodeURIComponent(question) + "&session=" + encodeURIComponent(sessionId)
  );

  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "progress") {
      setProgress(assistantEl, data.message);
    } else if (data.type === "answer") {
      setAnswer(assistantEl, data);
      source.close();
      setBusy(false);
    } else if (data.type === "error") {
      setError(assistantEl, data.message);
      source.close();
      setBusy(false);
    }
  };

  source.onerror = () => {
    if (assistantEl.querySelector(".bubble").classList.contains("progress")) {
      setError(assistantEl, "connection lost");
    }
    source.close();
    setBusy(false);
  };
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  input.style.height = "auto";
  ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});
