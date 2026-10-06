"""
Local GPT - a ChatGPT-style chat UI for Ollama, built with Streamlit.

Runs 100% offline: the only network call is to your own Ollama server
(default http://localhost:11434). Chat history is stored in a local SQLite file.

Run:
    streamlit run app.py

Optional environment variables:
    OLLAMA_HOST    Ollama server address      (default: http://localhost:11434)
    APP_PASSWORD   Shared password gate       (default: disabled)
    CHAT_DB_PATH   SQLite file for history    (default: ./chats.db)
"""

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
APP_TITLE = "Local GPT"
DB_PATH = Path(os.environ.get("CHAT_DB_PATH", Path(__file__).with_name("chats.db")))
MAX_DOC_CHARS = 100_000
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant running privately inside our "
    "organization. Answer clearly and concisely. If you are unsure, say so."
)


def normalize_host(host: str) -> str:
    host = (host or "").strip().rstrip("/")
    if not host:
        host = "http://localhost:11434"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


DEFAULT_HOST = normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))


# --------------------------------------------------------------------------- #
# Storage (SQLite)
# --------------------------------------------------------------------------- #
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
            """
        )


def list_chats() -> list[sqlite3.Row]:
    with db() as conn:
        return conn.execute("SELECT id, title FROM chats ORDER BY created DESC").fetchall()


def create_chat(title: str) -> str:
    chat_id = uuid.uuid4().hex
    with db() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, created) VALUES (?, ?, ?)",
            (chat_id, title, time.time()),
        )
    return chat_id


def delete_chat(chat_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def load_messages(chat_id: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def save_message(chat_id: str, role: str, content: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, time.time()),
        )


def make_title(text: str) -> str:
    text = " ".join(text.split())
    return (text[:40] + "…") if len(text) > 40 else (text or "New chat")


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=10, show_spinner=False)
def fetch_models(host: str) -> tuple[list[str], str | None]:
    """Return (model_names, error_message)."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        names = sorted(m["name"] for m in r.json().get("models", []))
        return names, None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


def stream_chat(host: str, model: str, messages: list[dict], options: dict):
    """Yield text chunks from Ollama's streaming /api/chat endpoint."""
    payload = {"model": model, "messages": messages, "stream": True, "options": options}
    with requests.post(
        f"{host}/api/chat", json=payload, stream=True, timeout=(5, 900)
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def password_gate() -> bool:
    expected = os.environ.get("APP_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("authed"):
        return True
    st.title(f"🔒 {APP_TITLE}")
    pwd = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if pwd == expected:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def chat_to_markdown(title: str, messages: list[dict]) -> str:
    lines = [f"# {title}", ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines += [f"**{who}:**", "", m["content"], ""]
    return "\n".join(lines)


def render_sidebar() -> dict:
    """Draw the sidebar and return the current generation settings."""
    with st.sidebar:
        st.title(f"🤖 {APP_TITLE}")

        if st.button("➕ New chat", use_container_width=True, type="primary"):
            st.session_state.chat_id = None
            st.session_state.doc_text = ""
            st.session_state.doc_name = ""
            st.rerun()

        # ---- Chat history ----
        st.caption("Chats")
        chats = list_chats()
        if not chats:
            st.write("_No chats yet_")
        for c in chats:
            col_open, col_del = st.columns([5, 1])
            is_active = c["id"] == st.session_state.chat_id
            if col_open.button(
                c["title"],
                key=f"open_{c['id']}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
            ):
                st.session_state.chat_id = c["id"]
                st.rerun()
            if col_del.button("🗑", key=f"del_{c['id']}", help="Delete chat"):
                delete_chat(c["id"])
                if is_active:
                    st.session_state.chat_id = None
                st.rerun()

        st.divider()

        # ---- Model & connection ----
        with st.expander("Model settings", expanded=True):
            host = normalize_host(
                st.text_input("Ollama server", value=DEFAULT_HOST, key="host")
            )
            models, err = fetch_models(host)
            if err:
                st.error(f"Can't reach Ollama at {host}\n\n{err}")
            elif not models:
                st.warning("Connected, but no models installed. Run `ollama pull <model>`.")

            model = st.selectbox("Model", models, index=0 if models else None)
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
            num_ctx = st.select_slider(
                "Context window (tokens)",
                options=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
                value=8192,
            )
            system_prompt = st.text_area(
                "System prompt", value=DEFAULT_SYSTEM_PROMPT, height=120
            )

        # ---- Document context ----
        with st.expander("Attach a document"):
            st.caption("Text-based files only (txt, md, csv, json, code, logs).")
            up = st.file_uploader(
                "Document",
                type=["txt", "md", "csv", "json", "py", "js", "ts", "html", "log", "yaml", "yml", "sql"],
                label_visibility="collapsed",
            )
            if up is not None:
                text = up.getvalue().decode("utf-8", errors="replace")
                if len(text) > MAX_DOC_CHARS:
                    st.warning(f"File truncated to {MAX_DOC_CHARS:,} characters.")
                    text = text[:MAX_DOC_CHARS]
                st.session_state.doc_text = text
                st.session_state.doc_name = up.name
            if st.session_state.doc_text:
                st.success(f"Using: {st.session_state.doc_name}")
                if st.button("Remove document"):
                    st.session_state.doc_text = ""
                    st.session_state.doc_name = ""
                    st.rerun()

    return {
        "host": host,
        "model": model,
        "options": {"temperature": temperature, "num_ctx": int(num_ctx)},
        "system_prompt": system_prompt.strip(),
    }


def build_messages(history: list[dict], system_prompt: str) -> list[dict]:
    system = system_prompt
    if st.session_state.doc_text:
        system += (
            f"\n\nThe user attached a document named '{st.session_state.doc_name}'. "
            "Use it as context when relevant.\n\n--- DOCUMENT START ---\n"
            f"{st.session_state.doc_text}\n--- DOCUMENT END ---"
        )
    msgs = [{"role": "system", "content": system}] if system.strip() else []
    return msgs + history


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
    init_db()

    st.session_state.setdefault("chat_id", None)
    st.session_state.setdefault("doc_text", "")
    st.session_state.setdefault("doc_name", "")

    if not password_gate():
        return

    cfg = render_sidebar()

    chat_id = st.session_state.chat_id
    history = load_messages(chat_id) if chat_id else []

    # Header
    st.header("How can I help you today?" if not history else "Chat")

    # Existing messages
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Export button (only when there is something to export)
    if history:
        title = next((c["title"] for c in list_chats() if c["id"] == chat_id), "Chat")
        st.sidebar.download_button(
            "⬇️ Export chat (.md)",
            data=chat_to_markdown(title, history),
            file_name="chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # New input
    prompt = st.chat_input("Message your local model…", disabled=not cfg["model"])
    if not prompt:
        return

    is_new_chat = chat_id is None
    if is_new_chat:
        chat_id = create_chat(make_title(prompt))
        st.session_state.chat_id = chat_id

    save_message(chat_id, "user", prompt)
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                stream_chat(
                    cfg["host"],
                    cfg["model"],
                    build_messages(history, cfg["system_prompt"]),
                    cfg["options"],
                )
            )
        except requests.exceptions.ConnectionError:
            reply = None
            st.error(f"Could not connect to Ollama at {cfg['host']}. Is `ollama serve` running?")
        except Exception as exc:  # noqa: BLE001
            reply = None
            st.error(f"Generation failed: {exc}")

    if reply:
        save_message(chat_id, "assistant", reply)

    if is_new_chat:
        st.rerun()  # refresh sidebar so the new chat appears in the list


if __name__ == "__main__":
    main()
