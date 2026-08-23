"""Document parser and text chunker for Markdown and PDF files using pymupdf4llm."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import pymupdf4llm

from ..core.logger import logger


@dataclass
class DocumentChunk:
    """Represents a text chunk ready for vector embedding."""
    content: str
    metadata: Dict[str, Any]


class DocumentParser:
    """Parses raw PDF and Markdown documents into chunked documents with table-awareness."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """Cleans and normalizes text content."""
        if not text:
            return ""
        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Normalize whitespace while preserving line structure
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def clean_pdf_markdown(self, text: str) -> str:
        """Cleans PDF-specific artifacts (page numbers, split tables, repeated headers)."""
        if not text:
            return ""

        # Remove standalone page numbers (e.g. 9-19-1, 12-22, Page 1)
        text = re.sub(r"\n\s*\d+-\d+(-\d+)?\s*\n", "\n\n", text)
        text = re.sub(r"\n\s*Page \d+\s*\n", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\n\s*-\s*\d+\s*-\s*\n", "\n\n", text)

        # Stitch broken markdown tables caused by page breaks
        # Pattern: table row, followed by newlines and another table row or delimiter
        text = re.sub(r"(\|[^\n]+\|)\s*\n\s*(\|[^\n]+\|)", r"\1\n\2", text)

        return self.clean_text(text)

    def _split_table_rows(
        self, table_text: str, header_rows: str, title_prefix: str, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Splits an oversized Markdown table into row chunks while preserving header context."""
        lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
        if len(lines) <= 2:
            content = f"{title_prefix}{table_text}".strip()
            return [DocumentChunk(content=content, metadata=dict(metadata))]

        # lines[0] = header, lines[1] = delimiter, lines[2:] = data rows
        data_rows = lines[2:]
        chunks: List[DocumentChunk] = []
        current_rows: List[str] = []
        current_length = len(title_prefix) + len(header_rows)

        for row in data_rows:
            row_len = len(row) + 1
            if current_rows and (current_length + row_len > self.chunk_size):
                chunk_str = f"{title_prefix}{header_rows}\n" + "\n".join(current_rows)
                chunks.append(DocumentChunk(content=chunk_str.strip(), metadata=dict(metadata)))
                current_rows = []
                current_length = len(title_prefix) + len(header_rows)

            current_rows.append(row)
            current_length += row_len

        if current_rows:
            chunk_str = f"{title_prefix}{header_rows}\n" + "\n".join(current_rows)
            chunks.append(DocumentChunk(content=chunk_str.strip(), metadata=dict(metadata)))

        return chunks

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Table-aware text chunking that preserves markdown table integrity and headers."""
        text = self.clean_text(text)
        if not text:
            return []

        title = metadata.get("title", "")
        title_prefix = f"【來源規章: {title}】\n\n" if title else ""

        # Identify markdown tables and paragraphs
        # A markdown table is a block of consecutive lines starting with '|'
        lines = text.split("\n")
        blocks: List[str] = []
        current_block: List[str] = []
        is_table = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                if not is_table:
                    if current_block:
                        blocks.append("\n".join(current_block))
                        current_block = []
                    is_table = True
                current_block.append(line)
            else:
                if is_table:
                    if current_block:
                        blocks.append("\n".join(current_block))
                        current_block = []
                    is_table = False
                current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))

        chunks: List[DocumentChunk] = []
        current_text = ""

        for block in blocks:
            stripped_block = block.strip()
            if not stripped_block:
                continue

            # Check if block is a markdown table
            is_table_block = all(
                l.strip().startswith("|") and l.strip().endswith("|")
                for l in stripped_block.split("\n")
                if l.strip()
            )

            if is_table_block:
                table_lines = [l.strip() for l in stripped_block.split("\n") if l.strip()]
                if len(table_lines) >= 2 and "|---" in table_lines[1]:
                    header_rows = f"{table_lines[0]}\n{table_lines[1]}"
                else:
                    header_rows = ""

                # If current_text exists, flush it before handling table
                if current_text.strip():
                    full_content = f"{title_prefix}{current_text.strip()}"
                    chunks.append(DocumentChunk(content=full_content, metadata=dict(metadata)))
                    current_text = ""

                # If table fits inside chunk size
                if len(title_prefix) + len(stripped_block) <= self.chunk_size:
                    full_content = f"{title_prefix}{stripped_block}"
                    chunks.append(DocumentChunk(content=full_content, metadata=dict(metadata)))
                else:
                    # Oversized table: split across rows with header preserved
                    table_chunks = self._split_table_rows(
                        stripped_block, header_rows, title_prefix, metadata
                    )
                    chunks.extend(table_chunks)
            else:
                # Regular text block
                if current_text and (len(current_text) + len(stripped_block) + 2 > self.chunk_size):
                    full_content = f"{title_prefix}{current_text.strip()}"
                    chunks.append(DocumentChunk(content=full_content, metadata=dict(metadata)))
                    current_text = ""

                current_text += ("\n\n" if current_text else "") + stripped_block

        if current_text.strip():
            full_content = f"{title_prefix}{current_text.strip()}"
            chunks.append(DocumentChunk(content=full_content, metadata=dict(metadata)))

        # Assign chunk indices
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks

    def parse_markdown_file(self, file_path: Path) -> List[DocumentChunk]:
        """Parses a Markdown file and returns chunks."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            title = file_path.stem
            metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "title": title,
                "doc_type": "markdown",
            }
            chunks = self.chunk_text(content, metadata)
            logger.info(f"Parsed Markdown: {file_path.name} -> {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to parse markdown {file_path}: {e}")
            return []

    def parse_pdf_file(self, file_path: Path) -> List[DocumentChunk]:
        """Parses a PDF file using pymupdf4llm with table cleaning and returns chunks."""
        try:
            raw_md = pymupdf4llm.to_markdown(str(file_path))
            cleaned_md = self.clean_pdf_markdown(raw_md)
            title = file_path.stem
            metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "title": title,
                "doc_type": "pdf",
            }
            chunks = self.chunk_text(cleaned_md, metadata)
            logger.info(f"Parsed PDF (pymupdf4llm): {file_path.name} -> {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path} with pymupdf4llm: {e}", exc_info=True)
            return []

    def parse_directory(self, raw_dir: Path, markdown_dir: Path) -> List[DocumentChunk]:
        """Parses all markdown and PDF files in given directories."""
        all_chunks: List[DocumentChunk] = []

        # Parse markdown files
        if markdown_dir.exists():
            for md_file in markdown_dir.glob("*.md"):
                all_chunks.extend(self.parse_markdown_file(md_file))

        # Parse PDF files
        if raw_dir.exists():
            for pdf_file in raw_dir.glob("*.pdf"):
                all_chunks.extend(self.parse_pdf_file(pdf_file))

        logger.info(f"Total chunks extracted across all sources: {len(all_chunks)}")
        return all_chunks
