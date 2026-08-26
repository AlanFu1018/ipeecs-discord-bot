"""Document parser and text chunker for Markdown, PDF, and DOCX files."""
import re
import xml.etree.ElementTree as ET
import zipfile
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
    """Parses Markdown, PDF, and DOCX documents into chunked documents."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100, chunk_mini: int = 100):
        self.chunk_size = chunk_size
        self.chunk_mini = chunk_mini
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

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Splits a single text into overlapping chunks."""
        text = self.clean_text(text)
        if not text:
            return []

        # Divide all text and tables
        text_length = len(text)
        slices: List[str] = []
        checker_start = 0
        checker_end = 0
        if '|' not in text:  # If there's ain't any table, no need of slicing
            checker_end = text_length
        while checker_end < text_length:
            if text[checker_end] == '|':
                checker_end = text.rfind('#', checker_start, checker_end) + 1
                if text[checker_start:checker_end]:
                    slices.append(text[checker_start:checker_end])
                checker_start = checker_end
                checker_end = (
                    text.find('#', checker_start, text_length)) \
                    if text.find('#', checker_start, text_length) != -1 \
                    else text_length
                slices.append(text[checker_start:checker_end])
                checker_start = checker_end

            checker_end += 1
        if text[checker_start:text_length]:
            slices.append(text[checker_start:text_length])

        # Dealing with categorized dic containing text and table
        chunk_index = 0
        chunks: List[DocumentChunk] = []
        for sli in slices:
            if '|' in sli:
                meta = dict(metadata)
                chunk_str = '#' + meta["title"] + '\n' + sli
                chunk_str = chunk_str.strip()
                meta["chunk_index"] = chunk_index
                chunks.append(DocumentChunk(content=chunk_str, metadata=meta))
                chunk_index += 1
            else:
                start = 0
                sli_len = len(sli)
                while start < sli_len:
                    end = start + self.chunk_size

                    if end >= sli_len:
                        chunk_str = sli[start:]
                    else:
                        sub = sli[start:end]
                        # Try to break at natural boundary (newline, period, question mark, semicolon)
                        cut = max(
                            sub.rfind("\n"),
                            sub.rfind("。"),
                            sub.rfind("！"),
                            sub.rfind("？"),
                            sub.rfind(". "),
                        )
                        if cut > self.chunk_size // 2:
                            end = start + cut + 1
                        chunk_str = sli[start:end]

                    if len(chunk_str) > self.chunk_mini:  # Ignore tiny noisy chunks
                        meta = dict(metadata)
                        chunk_str = '#' + meta["title"] + '\n' + chunk_str
                        chunk_str = chunk_str.strip()
                        meta["chunk_index"] = chunk_index
                        chunks.append(DocumentChunk(content=chunk_str, metadata=meta))
                        chunk_index += 1

                    if end >= sli_len:
                        break
                    start = end - self.chunk_overlap

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
        """Parses a PDF file using pymupdf4llm and returns chunks."""
        try:
            md_content = pymupdf4llm.to_markdown(str(file_path))
            title = file_path.stem
            metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "title": title,
                "doc_type": "pdf",
            }
            chunks = self.chunk_text(md_content, metadata)
            logger.info(f"Parsed PDF (pymupdf4llm): {file_path.name} -> {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path} with pymupdf4llm: {e}", exc_info=True)
            return []

    def parse_docx_file(self, file_path: Path) -> List[DocumentChunk]:
        """Parses a DOCX file and returns chunks."""
        try:
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            with zipfile.ZipFile(file_path) as z:
                xml_content = z.read("word/document.xml")

            root = ET.fromstring(xml_content)
            body = root.find("w:body", ns)
            if body is None:
                body = root

            lines: List[str] = []
            for child in body:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag == "p":
                    text_parts = [node.text for node in child.findall(".//w:t", ns) if node.text]
                    para_text = "".join(text_parts).strip()
                    if para_text:
                        lines.append(para_text)
                elif tag == "tbl":
                    for tr in child.findall("w:tr", ns):
                        cells = []
                        for tc in tr.findall("w:tc", ns):
                            cell_text = "".join(node.text for node in tc.findall(".//w:t", ns) if node.text).strip()
                            cells.append(cell_text)
                        if cells:
                            lines.append(" | ".join(cells))

            content = "\n\n".join(lines)
            title = file_path.stem
            metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "title": title,
                "doc_type": "docx",
            }
            chunks = self.chunk_text(content, metadata)
            logger.info(f"Parsed DOCX: {file_path.name} -> {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to parse DOCX {file_path}: {e}", exc_info=True)
            return []

    def parse_directory(self, raw_dir: Path, markdown_dir: Path) -> List[DocumentChunk]:
        """Parses all Markdown files, text-dominant PDF files, and DOCX files."""
        all_chunks: List[DocumentChunk] = []

        # 1. Parse all Markdown files (scraped web pages + Gemini-converted table PDFs)
        if markdown_dir.exists():
            for md_file in sorted(markdown_dir.glob("*.md")):
                all_chunks.extend(self.parse_markdown_file(md_file))

        # 2. Parse text-dominant PDF and DOCX files (e.g. 國立中央大學學則, 資工系會議室教室教學實驗室管理細則)
        text_pdf_dir = raw_dir / "text_pdfs"
        search_dirs = [text_pdf_dir] if (text_pdf_dir.exists() and (list(text_pdf_dir.glob("*.pdf")) or list(text_pdf_dir.glob("*.docx")))) else [raw_dir]

        for s_dir in search_dirs:
            if s_dir.exists():
                for pdf_file in sorted(s_dir.glob("*.pdf")):
                    all_chunks.extend(self.parse_pdf_file(pdf_file))
                for docx_file in sorted(s_dir.glob("*.docx")):
                    all_chunks.extend(self.parse_docx_file(docx_file))

        logger.info(f"Total chunks extracted across all sources: {len(all_chunks)}")
        return all_chunks
