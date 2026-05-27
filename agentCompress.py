"""
Context-generation agent with automatic compression and disk persistence.

Run:
  .\\.venv\\Scripts\\python.exe agentCompress.py
  .\\.venv\\Scripts\\python.exe agentCompress.py --no-load

On startup, the latest file in compressed_context/ is auto-loaded when
AUTO_LOAD_SAVED_CONTEXT is True. Set SAVED_CONTEXT_FILE to load a specific file.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---

DEEPSEEK_API_KEY = "sk-b818de2caa644d908f15e51241e79f48"
DEEPSEEK_MODEL = "deepseek-chat"
COMPRESSED_CONTEXT_DIR = "compressed_context"
COMPRESS_THRESHOLD_CHARS = 4000
GENERATION_ROUNDS = 4
DEFAULT_TOPIC = "LLM agents: planning, memory, and tool use"
AUTO_LOAD_SAVED_CONTEXT = True
SAVED_CONTEXT_FILE = None

os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

GENERATION_SYSTEM_PROMPT = (
    "You are a research agent that builds rich factual context on a topic. "
    "Each turn should add new details: definitions, methods, trade-offs, and examples. "
    "Do not repeat prior content verbatim; extend and deepen the knowledge base."
)

COMPRESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You compress long research context into a dense summary. "
            "Preserve facts, terminology, relationships, and actionable points. "
            "Remove redundancy and filler. Use clear sections or bullet points.",
        ),
        ("human", "Compress the following context:\n\n{context}"),
    ]
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def build_llm() -> BaseChatModel:
    return init_chat_model(
        DEEPSEEK_MODEL,
        model_provider="deepseek",
        temperature=0.7,
        max_tokens=2048,
    )


def build_generation_agent(llm: BaseChatModel):
    return create_agent(llm, tools=[], system_prompt=GENERATION_SYSTEM_PROMPT)


def _last_assistant_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def compress_context(llm: BaseChatModel, context: str) -> str:
    chain = COMPRESS_PROMPT | llm
    response = chain.invoke({"context": context})
    return response.content if isinstance(response.content, str) else str(response.content)


def save_compressed_context(
    compressed: str,
    *,
    topic: str,
    original_chars: int,
    round_number: int,
    output_dir: str = COMPRESSED_CONTEXT_DIR,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    text_path = out_dir / f"context_{stamp}.txt"
    meta_path = out_dir / f"context_{stamp}_meta.json"

    text_path.write_text(compressed, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "topic": topic,
                "round_number": round_number,
                "original_chars": original_chars,
                "compressed_chars": len(compressed),
                "compression_ratio": round(len(compressed) / original_chars, 4)
                if original_chars
                else 0,
                "saved_at_utc": stamp,
                "text_file": text_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return text_path


def list_saved_context_files(output_dir: str = COMPRESSED_CONTEXT_DIR) -> list[Path]:
    """Return saved context .txt files, newest first."""
    out_dir = Path(output_dir)
    if not out_dir.is_dir():
        return []
    files = sorted(
        out_dir.glob("context_*.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files


def find_latest_saved_context(output_dir: str = COMPRESSED_CONTEXT_DIR) -> Path | None:
    """Return the most recently saved context file, or None."""
    files = list_saved_context_files(output_dir)
    return files[0] if files else None


def _meta_path_for(text_path: Path) -> Path:
    return text_path.with_name(f"{text_path.stem}_meta.json")


def load_saved_context(
    path: Path | str | None = None,
    output_dir: str = COMPRESSED_CONTEXT_DIR,
) -> tuple[str, Path, dict | None]:
    """
    Load compressed context from disk.

    If path is None, loads the newest file in output_dir.
    Returns (context_text, file_path, metadata_or_none).
    """
    if path is None:
        resolved = find_latest_saved_context(output_dir)
        if resolved is None:
            raise FileNotFoundError(
                f"No saved context files found in {Path(output_dir).resolve()}"
            )
    else:
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Context file not found: {resolved.resolve()}")

    context_text = resolved.read_text(encoding="utf-8")
    meta_path = _meta_path_for(resolved)
    metadata = None
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    return context_text, resolved, metadata


class ContextCompressionAgent:
    """Generates context in rounds, compresses at threshold, saves to folder."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        *,
        compress_threshold: int = COMPRESS_THRESHOLD_CHARS,
        output_dir: str = COMPRESSED_CONTEXT_DIR,
        auto_load: bool = AUTO_LOAD_SAVED_CONTEXT,
    ):
        self.llm = llm or build_llm()
        self.compress_threshold = compress_threshold
        self.output_dir = output_dir
        self.agent = build_generation_agent(self.llm)
        self.context_blocks: list[str] = []
        self.loaded_from: Path | None = None
        self.loaded_topic: str | None = None

        if auto_load:
            self.try_load_saved_context()

    @property
    def full_context(self) -> str:
        return "\n".join(self.context_blocks)

    def try_load_saved_context(
        self,
        path: Path | str | None = None,
    ) -> Path | None:
        """Load saved context if available; return path or None."""
        try:
            load_path = path or SAVED_CONTEXT_FILE
            context_text, resolved, metadata = load_saved_context(
                load_path,
                output_dir=self.output_dir,
            )
        except FileNotFoundError:
            return None

        self.loaded_from = resolved
        self.loaded_topic = metadata.get("topic") if metadata else None
        self.context_blocks = [
            f"--- Loaded from {resolved.name} ---\n{context_text}"
        ]
        return resolved

    def generate_round(self, topic: str, round_number: int) -> str:
        prior = self.full_context or "(none yet)"
        user_message = (
            f"Topic: {topic}\n"
            f"Round: {round_number}\n\n"
            f"Prior accumulated context:\n{prior}\n\n"
            "Add the next layer of research context for this topic."
        )
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
        text = _last_assistant_text(result["messages"])
        self.context_blocks.append(f"--- Round {round_number} ---\n{text}")
        return text

    def maybe_compress_and_save(self, topic: str, round_number: int) -> Path | None:
        full = self.full_context
        if len(full) < self.compress_threshold:
            return None

        _log(
            f"Context size {len(full)} chars >= threshold "
            f"{self.compress_threshold}; compressing..."
        )
        compressed = compress_context(self.llm, full)
        path = save_compressed_context(
            compressed,
            topic=topic,
            original_chars=len(full),
            round_number=round_number,
            output_dir=self.output_dir,
        )

        self.context_blocks = [f"--- Compressed summary (round {round_number}) ---\n{compressed}"]
        _log(f"Saved compressed context to {path}")
        return path

    def run(self, topic: str = DEFAULT_TOPIC, rounds: int = GENERATION_ROUNDS) -> list[Path]:
        saved_paths: list[Path] = []

        for round_number in range(1, rounds + 1):
            _log(f"\nGenerating context — round {round_number}/{rounds}...")
            self.generate_round(topic, round_number)
            _log(f"Buffer size: {len(self.full_context)} chars")

            saved = self.maybe_compress_and_save(topic, round_number)
            if saved:
                saved_paths.append(saved)

        if self.full_context.strip() and (
            not saved_paths or len(self.full_context) >= self.compress_threshold
        ):
            _log("\nFinal compression pass...")
            compressed = compress_context(self.llm, self.full_context)
            path = save_compressed_context(
                compressed,
                topic=topic,
                original_chars=len(self.full_context),
                round_number=rounds,
                output_dir=self.output_dir,
            )
            saved_paths.append(path)
            _log(f"Saved final compressed context to {path}")

        return saved_paths


def main():
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-deepseek-api-key-here":
        _log("Error: set DEEPSEEK_API_KEY in the configuration section.")
        sys.exit(1)

    auto_load = AUTO_LOAD_SAVED_CONTEXT and "--no-load" not in sys.argv
    _log(f"Output folder: {Path(COMPRESSED_CONTEXT_DIR).resolve()}")

    agent = ContextCompressionAgent(auto_load=auto_load)
    topic = agent.loaded_topic or DEFAULT_TOPIC

    if agent.loaded_from:
        _log(
            f"Auto-loaded context from {agent.loaded_from.name} "
            f"({len(agent.full_context)} chars)"
        )
        if agent.loaded_topic:
            _log(f"Resuming topic from metadata: {agent.loaded_topic}")
    elif auto_load:
        _log("No saved context found; starting with an empty buffer.")

    saved = agent.run(topic=topic)

    if not saved:
        _log("No new files saved (context never reached compression threshold).")
    else:
        _log(f"\nDone. {len(saved)} compressed file(s) written.")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        _log(f"\nMissing dependency: {exc}")
        _log("Install with: .venv\\Scripts\\pip install -r requirements.txt")
        sys.exit(1)
