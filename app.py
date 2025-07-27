"""
Streamlit PDF/URL Analyser + Chatbot
===================================

A self-contained Streamlit application that

1. Indexes arbitrary PDFs and/or web pages.  
2. Auto-generates *n* sample Question-Answer pairs per data source.  
3. Lets the user chat with the indexed content (basic RAG workflow).

"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────
import io
import random
import re
from typing import List, Tuple
from urllib.parse import urlparse

# ── third-party ───────────────────────────────────────────────────────────
import requests
import PyPDF2
import faiss
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


class ChatbotApp:
    """Bundles helpers + Streamlit UI."""

    # ── configuration ──────────────────────────────────────────────────────
    EMBED_MODEL = "all-mpnet-base-v2"
    OPENROUTER_API_KEY = "[YOUR OPENROUTER_API_KEY]"
    OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
    MISTRAL_MODEL = "mistralai/mistral-7b-instruct:free"
    MAX_TOKENS = 700
    TEMPERATURE = 0.7

    CHUNK_SIZE = 800
    OVERLAP = 200
    TOP_K = 3
    SIM_THRESHOLD = 0.3
    OUT_OF_CONTEXT_REPLY = "I’m sorry, but I don’t have that information."

    # ── embedder cache ─────────────────────────────────────────────────────
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_embedder() -> SentenceTransformer:
        return SentenceTransformer(ChatbotApp.EMBED_MODEL)

    def __init__(self) -> None:
        self.embedder = self._load_embedder()

    # ── low-level extraction helpers ───────────────────────────────────────
    @staticmethod
    def extract_text_from_pdf(data: bytes) -> str:
        """Return concatenated text from a PDF byte-string."""
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    @staticmethod
    def extract_text_from_html(data: bytes) -> str:
        """Strip HTML tags and return visible text."""
        soup = BeautifulSoup(data, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "form"]):
            tag.decompose()
        return "\n".join(l.strip() for l in soup.get_text("\n").splitlines() if l.strip())

    @classmethod
    def chunk_text(cls, text: str) -> List[str]:
        """Split long text into overlapping word-chunks."""
        words, chunks, i = text.split(), [], 0
        while i < len(words):
            j = min(i + cls.CHUNK_SIZE, len(words))
            chunks.append(" ".join(words[i:j]))
            if j == len(words):
                break
            i += cls.CHUNK_SIZE - cls.OVERLAP
        return chunks

    # ── CHANGE ①: flexible OpenRouter wrapper ──────────────────────────────
    @classmethod
    def call_openrouter(cls, messages: list | str) -> str:
        """
        Accepts a ready-made message array **or** a legacy string prompt.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        resp = requests.post(
            cls.OPENROUTER_ENDPOINT,
            headers={
                "Authorization": f"Bearer {cls.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": cls.MISTRAL_MODEL,
                "messages": messages,
                "max_tokens": cls.MAX_TOKENS,
                "temperature": cls.TEMPERATURE,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ── mid-level helpers (unchanged) ───────────────────────────────────────
    @staticmethod
    def make_session_name(sources: List[str]) -> str:
        """Strips the titles of the PDF/URL nad keeps it as a name for the chat"""
        labels = []
        for src in sources:
            if src.startswith("http"):
                labels.append(urlparse(src).netloc.replace("www.", ""))
            else:
                labels.append(src.rsplit(".", 1)[0])
        uniq = []
        for l in labels:
            if l not in uniq:
                uniq.append(l)
            if len(uniq) == 3:
                break
        return " + ".join(uniq)

    @classmethod
    def qa_from_chunk(cls, chunk: str) -> Tuple[str, str]:
        """Return one Q-A pair generated from a text chunk."""
        prompt = (
            "You are a knowledgeable assistant. Based ONLY on the text below, "
            "create ONE question a user might ask and provide a short, factual answer. "
            "Respond in the exact format:\n\n"
            "Question: <question>\nAnswer: <answer>\n\n"
            f"TEXT:\n{chunk}\n"
        )
        raw = cls.call_openrouter(prompt)
        q, a = "—", "—"
        for line in raw.splitlines():
            if line.lower().startswith("question:"):
                q = line.split(":", 1)[1].strip()
            elif line.lower().startswith("answer:"):
                a = line.split(":", 1)[1].strip()
        return q, a

    @classmethod
    def build_sample_qas(
        cls, chunks: List[str], sources: List[str], n: int = 5
    ) -> List[dict]:
        """
        Generate ≤ *n* illustrative Q-A pairs, guaranteeing at least one per source.
        """
        qa_pairs: List[dict] = []
        # 1️⃣ Ensure each unique source contributes once
        seen: List[str] = []
        for idx, src in enumerate(sources):
            if src in seen:
                continue
            seen.append(src)
            q, a = cls.qa_from_chunk(chunks[idx])
            qa_pairs.append({"source": src, "q": q, "a": a})
            if len(qa_pairs) == n:
                return qa_pairs
        # 2️⃣ Fill remaining slots with random chunks
        remaining = n - len(qa_pairs)
        if remaining > 0:
            all_idxs = list(range(len(chunks)))
            random.shuffle(all_idxs)
            for idx in all_idxs:
                q, a = cls.qa_from_chunk(chunks[idx])
                qa_pairs.append({"source": sources[idx], "q": q, "a": a})
                if len(qa_pairs) == n:
                    break
        return qa_pairs

    # ─────────────────────────── Input/Extraction helpers ───────────────────
    @staticmethod
    def _gather_inputs(
        uploaded_files, url_text: str
    ) -> List[Tuple[str, bytes, str]]:
        """
        Collect PDF uploads *and* URLs into a unified list.

        Returns
        -------
        List[Tuple[str, bytes, str]]
            (source_label, raw_bytes, kind) with kind ∈ {"pdf", "html"}.
        """
        inputs: List[Tuple[str, bytes, str]] = []

        # PDFs
        for file in uploaded_files:
            try:
                inputs.append((file.name, file.read(), "pdf"))
            except (OSError, AttributeError) as err:
                st.warning(f"Could not read {file.name}: {err}")

        # URLs
        url_list = [u.strip() for u in re.split(r"[,\n]", url_text) if u.strip()]
        for single_url in url_list:
            try:
                response = requests.get(single_url, timeout=20)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                kind = (
                    "pdf"
                    if "application/pdf" in content_type
                    or single_url.lower().endswith(".pdf")
                    else "html"
                )
                inputs.append((single_url, response.content, kind))
            except requests.exceptions.RequestException as exc:
                st.error(f"URL error ({single_url}): {exc}")
        return inputs

    def _extract_chunks(
        self, inputs: List[Tuple[str, bytes, str]]
    ) -> Tuple[List[str], List[str]]:
        """
        Extract raw text and split into overlapping chunks.

        Returns
        -------
        Tuple[List[str], List[str]]
            (chunks, sources)
        """
        chunks: List[str] = []
        sources: List[str] = []
        for src, data, kind in inputs:
            text = (
                self.extract_text_from_pdf(data)
                if kind == "pdf"
                else self.extract_text_from_html(data)
            )
            if not text.strip():
                st.warning(f"No text found in {src}")
                continue
            for chunk in self.chunk_text(text):
                chunks.append(chunk)
                sources.append(src)
        return chunks, sources

    # ───────────────────────────── Streamlit UI layer ────────────────────────
    def run(self) -> None:
        """Top-level Streamlit orchestration."""
        if "sessions" not in st.session_state:
            st.session_state.sessions = []
        if "current" not in st.session_state:
            st.session_state.current = None

        # Sidebar
        st.sidebar.title("💬 Chats")
        options = ["➕ New Chat"] + [s["name"] for s in st.session_state.sessions]
        choice = st.sidebar.selectbox(
            "Your chats",
            options,
            index=0
            if st.session_state.current is None
            else st.session_state.current + 1,
        )
        st.session_state.current = None if choice == "➕ New Chat" else options.index(choice) - 1

        # Header
        st.title("PDF/URL Analyser and Chatbot")

        if st.session_state.current is None:
            self._index_step()
        else:
            self._chat_step()

    # ──────────────────────── UI sub-steps (private) ─────────────────────────
    def _index_step(self) -> None:
        """Upload, extract and create FAISS index"""

        st.subheader("Step 1: Add sources")

        with st.form("index_form", clear_on_submit=False):
            uploaded = st.file_uploader(
                "PDF file(s)", type="pdf", accept_multiple_files=True
            )
            urls_in = st.text_area(
                "Or paste one or more URLs (comma/new-line separated)"
            )
            submitted = st.form_submit_button("Process")

        if not submitted:   # no click → render form only
            return

        inputs = self._gather_inputs(uploaded, urls_in)
        if not inputs:
            st.error("Please add at least one PDF or valid URL.")
            return

        chunks, sources = self._extract_chunks(inputs)
        if not chunks:
            st.error("No text extracted from the provided sources.")
            return

        # Build FAISS index
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)            # pylint: disable=no-value-for-parameter
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)                     # pylint: disable=no-value-for-parameter

        # Auto-generate Q-A examples
        qas = self.build_sample_qas(chunks, sources, n=5)

        # Store the new session and switch to it
        st.session_state.sessions.append(
            {
                "name": self.make_session_name(sources),
                "chunks": chunks,
                "index": index,
                "history": [],
                "qas": qas,
            }
        )
        st.session_state.current = len(st.session_state.sessions) - 1

        # 🔄  Show the chat immediately (no second click needed)
        self._chat_step()

    # ------------------------------------------------------------------ #
    def _chat_step(self) -> None:
        """Render chat UI and answer user queries."""
        sess = st.session_state.sessions[st.session_state.current]
        st.subheader(f"Chat: {sess['name']}")

        if sess.get("qas"):
            st.markdown("### Sample Questions & Answers")
            for qa in sess["qas"]:
                st.markdown(f"**Q:** {qa['q']}\n\n**A:** {qa['a']}")
            st.markdown("---")

        for m in sess["history"]:
            st.chat_message(m["role"]).write(m["content"])

        user_q = st.chat_input("Ask anything about this document…")
        if not user_q:
            return

        sess["history"].append({"role": "user", "content": user_q})
        st.chat_message("user").write(user_q)

        q_emb = self.embedder.encode([user_q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        distances, indices = sess["index"].search(q_emb, self.TOP_K)

        if distances[0][0] < self.SIM_THRESHOLD:
            reply = self.OUT_OF_CONTEXT_REPLY
        else:
            keywords = {w.lower() for w in user_q.split() if len(w) > 3}
            top_chunks = [sess["chunks"][i] for i in indices[0]]
            if not any(any(k in c.lower() for k in keywords) for c in top_chunks):
                reply = self.OUT_OF_CONTEXT_REPLY
            else:
                # ── CHANGE ②: real system+user messages
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Answer in 150-200 words and "
                            "finish the last sentence completely."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Use ONLY the context below:\n\n"
                            + "\n\n---\n\n".join(top_chunks)
                            + f"\n\nQUESTION: {user_q}"
                        ),
                    },
                ]
                raw = self.call_openrouter(messages)
                reply = raw.strip()

        sess["history"].append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)

# ───────────────────────── entry point ────────────────────────────────────
if __name__ == "__main__":
    ChatbotApp().run()
