"""FastAPI backend for the chat UI.

Serves the static frontend (static/) and one streaming endpoint,
/api/chat, that runs src/pipeline.py's answer_question() and streams
progress + the final answer as Server-Sent Events -- so the UI can show
live pipeline stages ("Decomposing...", "Retrieving...", "Synthesizing...")
during the tens-of-seconds, multi-LLM-call round trip a compound question
takes, rather than a dead spinner.

The index (~110MB) and Retriever are loaded once at startup and reused
across every request, not reloaded per question.

Conversation history is kept server-side in `_sessions`, keyed by an opaque
`session` id the frontend generates once per page load (see static/app.js)
and passes as a query param -- EventSource is GET-only, so a small id in
the URL is the only option, not a full history blob. This is deliberately
not persisted anywhere (in-memory only, cleared on process restart, and
never written to disk) -- it survives a page reload only if the id itself
does, and static/app.js deliberately doesn't persist the id either, since
the visible chat transcript isn't persisted across reload anyway (see
src/contextualizer.py for what this history is actually used for).

answer_question() is a blocking, synchronous call chain (Ollama's client
is sync). Streaming its progress requires running it on a background
thread that pushes events onto a queue while a generator in the request
handler drains that queue and yields SSE frames as they arrive -- a plain
"call it and return the result" endpoint couldn't surface progress
mid-call.

Usage:
    python -m src.server
    (then open http://localhost:8000)
"""
from __future__ import annotations

import json
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.contextualizer import MAX_HISTORY_TURNS, Turn
from src.pipeline import answer_question
from src.retriever import Retriever

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_retriever: Retriever | None = None
_sessions: dict[str, list[Turn]] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _retriever
    _retriever = Retriever.load()
    yield


app = FastAPI(lifespan=_lifespan)


def _stream_answer(question: str, session_id: str | None):
    """Generator yielding SSE frames: progress events, then one answer (or error) event."""
    events: queue.Queue = queue.Queue()
    DONE = object()
    history = _sessions.setdefault(session_id, []) if session_id else []

    def worker() -> None:
        try:
            final = answer_question(
                question, _retriever, history=history,
                on_progress=lambda msg: events.put({"type": "progress", "message": msg}),
            )
            events.put({
                "type": "answer",
                "answer": final.answer,
                "fully_grounded": final.fully_grounded,
                "sources": final.sources,
                "intent": final.intent,
            })
            if session_id:
                history.append(Turn(question=question, answer=final.answer))
                del history[:-MAX_HISTORY_TURNS]  # nothing ever reads more than this
        except Exception as exc:
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put(DONE)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event = events.get()
        if event is DONE:
            break
        yield f"data: {json.dumps(event)}\n\n"


@app.get("/api/chat")
def chat(q: str, session: Optional[str] = None) -> StreamingResponse:
    # typing.Optional, not `str | None` -- FastAPI/Pydantic evaluates route
    # parameter annotations at request time (unlike plain internal
    # annotations elsewhere in this file), and PEP 604 `|` syntax under
    # `from __future__ import annotations` fails that evaluation on Python
    # 3.9 without the eval_type_backport package installed. Verified
    # directly: this 500ed on startup before the fix.
    return StreamingResponse(_stream_answer(q, session), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
