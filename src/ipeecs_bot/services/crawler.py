"""Web crawler and PDF downloader for Department regulations and FAQs."""
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urljoin, urlparse
import requests
import urllib3
import yaml
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

from ..core.logger import logger

# Disable SSL verification warnings for university websites with self-signed/internal certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DataCrawler:
    """Crawls department web pages to Markdown, downloads PDFs/DOCX, and converts table PDFs with Gemini."""

    def __init__(
        self,
        raw_dir: Path,
        markdown_dir: Path,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-3.1-flash-lite",
    ):
        self.raw_dir = Path(raw_dir)
        self.markdown_dir = Path(markdown_dir)
        self.text_pdf_dir = self.raw_dir / "text_pdfs"
        self.table_pdf_dir = self.raw_dir / "table_pdfs"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.text_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.table_pdf_dir.mkdir(parents=True, exist_ok=True)

        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or ""
        self.gemini_model = gemini_model
        self.genai_client = genai.Client(api_key=self.gemini_api_key) if self.gemini_api_key else None

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

    def fetch_url(self, url: str, timeout: int = 25) -> requests.Response:
        """Fetches page content with configured headers and SSL ignore."""
        resp = requests.get(url, headers=self.headers, timeout=timeout, verify=False)
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp

    def clean_text_content(self, soup: BeautifulSoup) -> str:
        """Extracts and formats main readable text into Markdown."""
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "iframe"]):
            tag.decompose()

        lines: List[str] = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "table"]):
            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue

            if element.name == "h1":
                lines.append(f"\n# {text}\n")
            elif element.name == "h2":
                lines.append(f"\n## {text}\n")
            elif element.name == "h3":
                lines.append(f"\n### {text}\n")
            elif element.name == "li":
                lines.append(f"- {text}")
            elif element.name == "p":
                lines.append(f"\n{text}\n")
            elif element.name == "tr":
                tds = [td.get_text(strip=True) for td in element.find_all(["td", "th"])]
                if tds:
                    lines.append(" | ".join(tds))

        content = "\n".join(lines)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        return content

    def download_pdf(self, pdf_url: str, save_name: str, target_dir: Optional[Path] = None) -> Optional[Path]:
        """Downloads a PDF or DOCX file and saves it with a sanitized filename into the target directory."""
        dest_dir = target_dir if target_dir else self.raw_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(pdf_url, headers=self.headers, timeout=30, verify=False)
            if resp.status_code == 200 and len(resp.content) > 0:
                clean_name = re.sub(r'[\\/*?:"<>|]', "_", save_name).strip()
                if not (clean_name.lower().endswith(".pdf") or clean_name.lower().endswith(".docx")):
                    content_type = resp.headers.get("Content-Type", "").lower()
                    if "word" in content_type or "docx" in content_type or "docx" in pdf_url.lower():
                        clean_name += ".docx"
                    else:
                        clean_name += ".pdf"

                file_path = dest_dir / clean_name
                with open(file_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"Downloaded file: {clean_name} ({len(resp.content):,} bytes) -> {dest_dir.name}/")
                return file_path
            else:
                logger.warning(f"Failed to download file (status {resp.status_code}): {pdf_url}")
        except Exception as e:
            logger.error(f"Error downloading file {pdf_url}: {e}")
        return None

    def crawl_markdown_page(self, title: str, url: str) -> Optional[Path]:
        """Crawls a single web page and saves clean content as a Markdown file."""
        logger.info(f"Crawling web page to Markdown: {title} -> {url}")
        try:
            resp = self.fetch_url(url)
            soup = BeautifulSoup(resp.text, "html.parser")

            page_title = soup.title.string.strip() if soup.title else title
            if title and title != url:
                page_title = title

            markdown_body = self.clean_text_content(soup)
            markdown_content = f"# {page_title}\n\n來源網址: {url}\n\n{markdown_body}\n"

            slug = re.sub(r'[\\/*?:"<>|]', "_", page_title).strip()
            md_path = self.markdown_dir / f"{slug}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            logger.info(f"Saved Markdown: {md_path.name}")
            return md_path
        except Exception as e:
            logger.error(f"Error crawling Markdown page {url}: {e}", exc_info=True)
            return None

    def crawl_academic_rules_pdf(self, rules_url: str) -> List[Path]:
        """Crawls National Central University Academic Rules (國立中央大學學則) page and downloads the PDF.

        Matches title="國立中央大學學則（PDF，另開新視窗）" or link text containing 學則 and PDF.
        """
        logger.info(f"Crawling NCU Academic Rules from: {rules_url}")
        downloaded: List[Path] = []
        try:
            resp = self.fetch_url(rules_url)
            soup = BeautifulSoup(resp.text, "html.parser")

            found_link = None
            for a in soup.find_all("a"):
                title_attr = a.get("title", "")
                text_content = a.get_text(strip=True)
                combined = f"{text_content} {title_attr}"

                if "國立中央大學學則" in combined and ("PDF" in combined or "pdf" in a.get("href", "").lower()):
                    found_link = a
                    break

            if found_link:
                href = found_link.get("href", "")
                if href and not href.startswith("javascript"):
                    pdf_url = urljoin(rules_url, href)
                    saved = self.download_pdf(
                        pdf_url,
                        "國立中央大學學則.pdf",
                        target_dir=self.text_pdf_dir,
                    )
                    if saved:
                        downloaded.append(saved)
            else:
                logger.warning(f"Could not locate 國立中央大學學則 PDF link on page: {rules_url}")

        except Exception as e:
            logger.error(f"Error crawling Academic Rules PDF: {e}", exc_info=True)

        return downloaded

    def crawl_csie_downloads(self, base_url: str, target_name: str = "資工系會議室教室教學實驗室管理細則") -> List[Path]:
        """Crawls CSIE downloads page (e.g. csie.ncu.edu.tw/downloads) and downloads target documents (DOCX/PDF)."""
        logger.info(f"Crawling CSIE downloads from: {base_url} for target: {target_name}")
        downloaded: List[Path] = []
        try:
            resp = self.fetch_url(base_url)
            soup = BeautifulSoup(resp.text, "html.parser")

            found_link = None
            save_filename = target_name

            for a in soup.find_all("a"):
                title_attr = a.get("title", "")
                text_content = a.get_text(strip=True)
                combined = f"{text_content} {title_attr}"

                # Match keywords in link or title
                if ("管理細則" in combined or "會議室" in combined or target_name in combined) and ("file" in a.get("href", "").lower() or "downloads" in base_url.lower()):
                    found_link = a
                    if ".docx" in combined or "docx" in combined.lower():
                        save_filename = target_name if target_name.endswith(".docx") else f"{target_name}.docx"
                    elif ".pdf" in combined or "pdf" in combined.lower():
                        save_filename = target_name if target_name.endswith(".pdf") else f"{target_name}.pdf"
                    break

            if found_link:
                href = found_link.get("href", "")
                if href and not href.startswith("javascript"):
                    file_url = urljoin(base_url, href)
                    saved = self.download_pdf(
                        file_url,
                        save_filename,
                        target_dir=self.text_pdf_dir,
                    )
                    if saved:
                        downloaded.append(saved)
            else:
                logger.warning(f"Could not locate '{target_name}' download link on page: {base_url}")

        except Exception as e:
            logger.error(f"Error crawling CSIE downloads {base_url}: {e}", exc_info=True)

        return downloaded

    def crawl_pdc_regulations(self, pdc_base_url: str) -> List[Path]:
        """Crawls PDC regulation directories and downloads target IPEECS table PDFs into table_pdf_dir."""
        logger.info(f"Starting PDC regulations crawl from: {pdc_base_url}")
        downloaded_pdfs: List[Path] = []
        target_keywords = [
            "資訊電機學院學士班",
            "電機工程專長",
            "資訊工程專長",
            "通訊工程專長",
            "網路工程專長",
        ]

        try:
            resp = self.fetch_url(pdc_base_url)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Step (1): Find links matching [0-9]{3}教務章則
            chapter_links: List[Tuple[str, str, str]] = []
            for a in soup.find_all("a"):
                text = a.get_text(strip=True)
                title_attr = a.get("title", "")
                combined_text = f"{text} {title_attr}".strip()

                year_match = re.search(r"(\d{3})\s*教務章則", combined_text)
                if year_match:
                    href = a.get("href", "")
                    if href and not href.startswith("javascript"):
                        full_chapter_url = urljoin(pdc_base_url, href)
                        year_str = year_match.group(1)
                        chapter_links.append((year_str, text or combined_text, full_chapter_url))

            logger.info(f"Found {len(chapter_links)} academic year regulation chapters on PDC.")

            # Step (2) & (3): For each chapter, navigate to 目次表 and download target PDFs
            for year, chap_title, chap_url in chapter_links:
                try:
                    chap_resp = self.fetch_url(chap_url)
                    chap_soup = BeautifulSoup(chap_resp.text, "html.parser")

                    mulu_urls: List[str] = []
                    for a in chap_soup.find_all("a"):
                        a_title = a.get("title", "")
                        a_text = a.get_text(strip=True)
                        if "國立中央大學各學士班應修科目及畢業條件目次表" in a_title or "國立中央大學各學士班應修科目及畢業條件目次表" in a_text:
                            href = a.get("href", "")
                            if href and not href.startswith("javascript"):
                                mulu_urls.append(urljoin(chap_url, href))

                    for mulu_url in mulu_urls:
                        mulu_resp = self.fetch_url(mulu_url)
                        mulu_soup = BeautifulSoup(mulu_resp.text, "html.parser")

                        for a in mulu_soup.find_all("a"):
                            t3 = a.get_text(strip=True)
                            title3 = a.get("title", "")
                            combined_item = f"{t3} {title3}".strip()
                            href3 = a.get("href", "")

                            if not href3 or href3.startswith("javascript"):
                                continue

                            for kw in target_keywords:
                                if kw in combined_item:
                                    pdf_url = urljoin(mulu_url, href3)

                                    # Build a clear descriptive filename
                                    item_label = t3 if (t3 and "新視窗" not in t3 and "開啟" not in t3) else title3
                                    if not item_label or "新視窗" in item_label:
                                        item_label = kw

                                    item_label = re.sub(r"[\r\n\t]+", "", item_label).strip()
                                    filename = f"{year}學年度_{item_label}"

                                    saved_path = self.download_pdf(
                                        pdf_url,
                                        filename,
                                        target_dir=self.table_pdf_dir,
                                    )
                                    if saved_path:
                                        downloaded_pdfs.append(saved_path)
                                    break
                except Exception as chap_err:
                    logger.error(f"Error processing chapter {year} ({chap_url}): {chap_err}")

        except Exception as e:
            logger.error(f"Error in PDC regulations crawler: {e}", exc_info=True)

        logger.info(f"PDC crawl completed. Downloaded {len(downloaded_pdfs)} regulation PDFs.")
        return downloaded_pdfs

    def crawl_course_regulation(self, course_base_url: str) -> List[Path]:
        """Crawls Course NCU and downloads 「創意與創業」學分學程選修辦法 PDF into table_pdf_dir."""
        logger.info(f"Starting Course NCU crawl from: {course_base_url}")
        downloaded_pdfs: List[Path] = []

        try:
            resp = self.fetch_url(course_base_url)
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a"):
                title_attr = a.get("title", "")
                text_content = a.get_text(strip=True)
                href = a.get("href", "")

                # Match title="「創意與創業」學分學程選修辦法(另開新視窗)"
                if ("創意與創業" in title_attr or "創意創業" in title_attr) and ("選修辦法" in title_attr or "辦法" in title_attr):
                    if href and not href.startswith("javascript"):
                        pdf_url = urljoin(course_base_url, href)
                        saved_path = self.download_pdf(
                            pdf_url,
                            "「創意與創業」學分學程選修辦法.pdf",
                            target_dir=self.table_pdf_dir,
                        )
                        if saved_path:
                            downloaded_pdfs.append(saved_path)
                elif ("創意與創業" in text_content or "創意創業" in text_content) and ("辦法" in text_content or "pdf" in href.lower()):
                    if href and not href.startswith("javascript"):
                        pdf_url = urljoin(course_base_url, href)
                        saved_path = self.download_pdf(
                            pdf_url,
                            "「創意與創業」學分學程選修辦法.pdf",
                            target_dir=self.table_pdf_dir,
                        )
                        if saved_path:
                            downloaded_pdfs.append(saved_path)

        except Exception as e:
            logger.error(f"Error crawling Course NCU regulation: {e}", exc_info=True)

        logger.info(f"Course NCU crawl completed. Downloaded {len(downloaded_pdfs)} PDFs.")
        return downloaded_pdfs

    def convert_pdf_to_markdown_gemini(
        self,
        pdf_path: Path,
        output_md_path: Path,
        max_retries: int = 5,
    ) -> Optional[Path]:
        """Sends a table-heavy PDF to Gemini to extract clean Markdown tables and rules."""
        if not self.genai_client:
            logger.warning(f"Gemini client not configured. Skipping Gemini conversion for {pdf_path.name}")
            return None

        prompt = (
            "你是一個專業的國立中央大學學術規章與學分學程表格整理專家。"
            "請將這份 PDF 文件完整轉換為結構清晰、語意完整且易於檢索的繁體中文 Markdown 格式。\n\n"
            "轉換要求：\n"
            "1. 完整保留並重現所有表格結構（使用標準 Markdown 表格語法 | 表頭1 | 表頭2 | ...）。\n"
            "2. 清晰列出所有學年度、專長名稱（如電機工程專長、資訊工程專長、通訊工程專長、網路工程專長或學士班）、"
            "必修與選修科目名稱、科目代碼、學分數以及畢業修業條件與學分要求規範。\n"
            "3. 完整保留所有附註、備註說明與各項規章要點，不要省略任何一條規則。\n"
            "4. 直接輸出完整的 Markdown 內容，不要包含多餘的聊天對話或開場白。"
        )

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            for attempt in range(1, max_retries + 1):
                try:
                    response = self.genai_client.models.generate_content(
                        model=self.gemini_model,
                        contents=[
                            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                            prompt,
                        ],
                    )

                    if response and response.text:
                        cleaned_text = response.text.strip()
                        if cleaned_text.startswith("```markdown"):
                            cleaned_text = cleaned_text[len("```markdown"):].strip()
                        elif cleaned_text.startswith("```"):
                            cleaned_text = cleaned_text[3:].strip()
                        if cleaned_text.endswith("```"):
                            cleaned_text = cleaned_text[:-3].strip()

                        with open(output_md_path, "w", encoding="utf-8") as f:
                            f.write(f"# {pdf_path.stem}\n\n{cleaned_text}\n")

                        logger.info(f"Gemini converted table PDF: {pdf_path.name} -> {output_md_path.name}")
                        return output_md_path
                    else:
                        logger.warning(f"Empty Gemini response for {pdf_path.name}")
                        break

                except Exception as api_err:
                    err_str = str(api_err)
                    if "429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait_time = attempt * 6.0
                        logger.warning(f"Rate limit / Busy ({err_str[:80]}). Waiting {wait_time}s before retry ({attempt}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Gemini API error on {pdf_path.name}: {api_err}")
                        break

        except Exception as e:
            logger.error(f"Failed to read/convert PDF {pdf_path}: {e}")

        return None

    def convert_all_table_pdfs(self, skip_converted: bool = False) -> List[Path]:
        """Converts all downloaded table PDFs in table_pdf_dir into Markdown files in markdown_dir.

        If skip_converted is True, skips converting PDFs whose corresponding markdown files already exist.
        """
        table_pdfs = list(self.table_pdf_dir.glob("*.pdf"))
        logger.info(f"Converting {len(table_pdfs)} table PDFs to Markdown via Gemini ({self.gemini_model})...")
        converted_mds: List[Path] = []

        for pdf_file in table_pdfs:
            md_filename = f"{pdf_file.stem}.md"
            out_md_path = self.markdown_dir / md_filename

            if skip_converted and out_md_path.exists() and out_md_path.stat().st_size > 0:
                logger.info(f"Skipping already converted table PDF: {pdf_file.name} (found {out_md_path.name})")
                converted_mds.append(out_md_path)
                continue

            res = self.convert_pdf_to_markdown_gemini(pdf_file, out_md_path)
            if res:
                converted_mds.append(res)
            # Sleep briefly between calls to stay well within Gemini API limits
            time.sleep(2.0)

        logger.info(f"Successfully converted/prepared {len(converted_mds)}/{len(table_pdfs)} table PDFs in Markdown.")
        return converted_mds

    def parse_urls_file(self, urls_file: Path) -> Dict[str, List[Tuple[str, str]]]:
        """Parses urls.yaml (or legacy urls.txt) into 3 sections: web, text_pdf, table_pdf."""
        sections: Dict[str, List[Tuple[str, str]]] = {
            "web": [],
            "text_pdf": [],
            "table_pdf": [],
        }

        if not urls_file.exists():
            logger.warning(f"URLs file not found: {urls_file}")
            return sections

        # Try YAML parsing first if extension is yaml/yml or regardless
        if urls_file.suffix.lower() in [".yaml", ".yml"]:
            try:
                with open(urls_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if isinstance(data, dict):
                    for sec in ["web", "text_pdf", "table_pdf"]:
                        items = data.get(sec, [])
                        if not isinstance(items, list):
                            continue
                        for item in items:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                sections[sec].append((str(item[0]).strip(), str(item[1]).strip()))
                            elif isinstance(item, (list, tuple)) and len(item) == 1:
                                u = str(item[0]).strip()
                                sections[sec].append((u, u))
                            elif isinstance(item, dict):
                                if "title" in item and "url" in item:
                                    sections[sec].append((str(item["title"]).strip(), str(item["url"]).strip()))
                                else:
                                    for k, v in item.items():
                                        sections[sec].append((str(k).strip(), str(v).strip()))
                            elif isinstance(item, str):
                                line = item.strip()
                                if "http://" in line or "https://" in line:
                                    if ":" in line and not line.startswith("http"):
                                        parts = line.split("http", 1)
                                        t = parts[0].strip().rstrip(":").strip()
                                        u = "http" + parts[1].strip()
                                    else:
                                        t = line
                                        u = line
                                    sections[sec].append((t, u))
                    return sections
            except Exception as e:
                logger.error(f"Error parsing YAML URLs file {urls_file}: {e}", exc_info=True)

        # Fallback to line-by-line parsing (for .txt files)
        try:
            with open(urls_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            current_section = "web"
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue

                lower_line = line.lower()
                if "//網站" in lower_line or "// 網站" in lower_line:
                    current_section = "web"
                    continue
                elif "文字為主" in lower_line:
                    current_section = "text_pdf"
                    continue
                elif "大量表格" in lower_line or "表格為主" in lower_line:
                    current_section = "table_pdf"
                    continue

                if line.startswith("//") or line.startswith("#"):
                    continue

                if "http://" in line or "https://" in line:
                    if ":" in line and not line.startswith("http"):
                        parts = line.split("http", 1)
                        title = parts[0].strip().rstrip(":").strip()
                        url = "http" + parts[1].strip()
                    else:
                        url = line.strip()
                        title = url

                    sections[current_section].append((title, url))
        except Exception as e:
            logger.error(f"Error reading URLs file {urls_file}: {e}", exc_info=True)

        return sections

    def crawl_all(
        self,
        urls_file: Path,
        skip_llm_convert: bool = False,
        skip_converted: bool = False,
    ) -> Dict[str, Any]:
        """Executes full 3-zone crawl based on urls.yaml."""
        logger.info(f"Parsing URLs config from: {urls_file}")
        targets = self.parse_urls_file(urls_file)

        results: Dict[str, Any] = {
            "markdown_files": [],
            "text_pdf_files": [],
            "table_pdf_files": [],
            "converted_table_markdowns": [],
        }

        # Zone 1: Crawl Web Pages directly to Markdown
        logger.info(f"--- [Zone 1] Crawling {len(targets['web'])} Web Pages to Markdown ---")
        for title, url in targets["web"]:
            md_path = self.crawl_markdown_page(title, url)
            if md_path:
                results["markdown_files"].append(md_path)

        # Zone 2: Crawl Text-Dominant PDFs and Documents
        logger.info(f"--- [Zone 2] Crawling {len(targets['text_pdf'])} Text-Dominant PDF/Doc Sources ---")
        for title, url in targets["text_pdf"]:
            if "pdc.adm.ncu.edu.tw" in url and ("1993" in url or "學則" in title):
                pdfs = self.crawl_academic_rules_pdf(url)
                results["text_pdf_files"].extend(pdfs)
            elif "csie.ncu.edu.tw" in url and ("downloads" in url or "管理細則" in title):
                files = self.crawl_csie_downloads(url, title)
                results["text_pdf_files"].extend(files)
            elif url.lower().endswith(".pdf") or url.lower().endswith(".docx"):
                saved = self.download_pdf(url, title, target_dir=self.text_pdf_dir)
                if saved:
                    results["text_pdf_files"].append(saved)
            else:
                # Generic fallback for text PDF / doc page
                pdfs = self.crawl_academic_rules_pdf(url)
                results["text_pdf_files"].extend(pdfs)

        # Zone 3: Crawl Table-Dominant PDFs & Convert to Markdown via Gemini
        logger.info(f"--- [Zone 3] Crawling {len(targets['table_pdf'])} Table-Dominant PDF Sources ---")
        for title, url in targets["table_pdf"]:
            if "pdc.adm.ncu.edu.tw" in url:
                pdfs = self.crawl_pdc_regulations(url)
                results["table_pdf_files"].extend(pdfs)
            elif "course.ncu.edu.tw" in url:
                pdfs = self.crawl_course_regulation(url)
                results["table_pdf_files"].extend(pdfs)
            elif url.lower().endswith(".pdf"):
                saved = self.download_pdf(url, title, target_dir=self.table_pdf_dir)
                if saved:
                    results["table_pdf_files"].append(saved)

        # Convert table PDFs to Markdown via Gemini
        if not skip_llm_convert:
            if results["table_pdf_files"] or list(self.table_pdf_dir.glob("*.pdf")):
                converted = self.convert_all_table_pdfs(skip_converted=skip_converted)
                results["converted_table_markdowns"].extend(converted)
        else:
            logger.info("Skipping Gemini table-to-markdown conversion (--skip-llm-convert enabled).")

        logger.info(
            f"=== Crawl & Conversion Finished ===\n"
            f"  - Web Markdowns: {len(results['markdown_files'])}\n"
            f"  - Text PDFs & Docs: {len(results['text_pdf_files'])}\n"
            f"  - Table PDFs Downloaded: {len(results['table_pdf_files'])}\n"
            f"  - Table PDFs Converted to Markdown: {len(results['converted_table_markdowns'])}\n"
        )
        return results
