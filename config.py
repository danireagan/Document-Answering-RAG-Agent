from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=1024, ge=1)

    # Embeddings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model",
    )
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device for local embeddings",
    )

    # Chunking
    # PDF: larger chunks work well (page-level text is dense prose)
    pdf_chunk_size: int = Field(default=512, ge=64, le=4096)
    pdf_chunk_overlap: int = Field(default=64, ge=0)
    # DOCX: smaller chunks work better (line-by-line structure, short lines)
    docx_chunk_size: int = Field(default=256, ge=64, le=4096)
    docx_chunk_overlap: int = Field(default=32, ge=0)

    # Retrieval
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0 = disabled)",
    )

    # ChromaDB
    chroma_persist_dir: Path = Field(
        default=Path("data/chroma_db"),
        description="ChromaDB persistence directory",
    )
    chroma_collection_name: str = Field(default="cv_chunks")

    # File Paths
    cv_dir: Path = Field(default=Path("data/cv"), description="Directory containing CV files")
    output_dir: Path = Field(default=Path("output"), description="Directory for output.txt")
    output_filename: str = Field(default="output.txt")
    log_dir: Path = Field(default=Path("logs"))
    log_filename: str = Field(default="app.log")

    # Logging
    log_level: str = Field(default="INFO", description="Python log level")
    log_to_file: bool = Field(default=True)
    log_to_console: bool = Field(default=True)

    # Agent Behaviour
    max_conversation_turns: int = Field(
        default=20,
        description=(
            "Max turns before history truncation. "
            "Each ReAct turn = 4 messages: "
            "HumanMessage + AIMessage(tool_call) + ToolMessage + AIMessage(answer)."
        ),
    )
    agent_max_iterations: int = Field(default=10, description="LangGraph recursion limit")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env")
        return self

    @property
    def output_file_path(self) -> Path:
        return self.output_dir / self.output_filename

    @property
    def log_file_path(self) -> Path:
        return self.log_dir / self.log_filename

    @property
    def numeric_log_level(self) -> int:
        return getattr(logging, self.log_level)

    @property
    def active_model(self) -> str:
        return self.openai_model


# Singleton — import this everywhere
settings = Settings()
