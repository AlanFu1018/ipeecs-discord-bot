"""Web crawler and PDF downloader for Department regulations and FAQs."""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urljoin, urlparse
import requests
import urllib3
from bs4 import BeautifulSoup

from ..core.logger import logger

# Disable SSL verification warnings for university websites with self-signed/internal certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DataCrawler:
    """Crawls department web pages to Markdown and downloads targeted regulation PDFs."""

    def __init__(self, raw_dir: Path, markdown_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.markdown_dir = Path(markdown_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

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

    def download_pdf(self, pdf_url: str, save_name: str) -> Optional[Path]:
        """Downloads a PDF file and saves it with a sanitized filename."""
        try:
            resp = requests.get(pdf_url, headers=self.headers, timeout=30, verify=False)
            if resp.status_code == 200 and len(resp.content) > 0:
                clean_name = re.sub(r'[\\/*?:"<>|]', "_", save_name).strip()
                if not clean_name.lower().endswith(".pdf"):
                    clean_name += ".pdf"

                pdf_path = self.raw_dir / clean_name
                with open(pdf_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"Downloaded PDF: {clean_name} ({len(resp.content):,} bytes)")
                return pdf_path
            else:
                logger.warning(f"Failed to download PDF (status {resp.status_code}): {pdf_url}")
        except Exception as e:
            logger.error(f"Error downloading PDF {pdf_url}: {e}")
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

    def crawl_pdc_regulations(self, pdc_base_url: str) -> List[Path]:
        """Crawls PDC regulation directories and downloads target IPEECS PDFs.

        1. Finds links matching `[0-9]{3}教務章則` (e.g. 114教務章則彙編)
        2. In each chapter page, finds link with title='國立中央大學各學士班應修科目及畢業條件目次表'
        3. In the 目次表 page, downloads PDFs matching keywords:
           - 資訊電機學院學士班
           - 電機工程專長
           - 資訊工程專長
           - 通訊工程專長
           - 網路工程專長
        """
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

                                    saved_path = self.download_pdf(pdf_url, filename)
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
        """Crawls Course NCU and downloads 「創意與創業」學分學程選修辦法 PDF."""
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
                if "創意與創業" in title_attr and "選修辦法" in title_attr:
                    if href and not href.startswith("javascript"):
                        pdf_url = urljoin(course_base_url, href)
                        saved_path = self.download_pdf(pdf_url, "「創意與創業」學分學程選修辦法.pdf")
                        if saved_path:
                            downloaded_pdfs.append(saved_path)
                elif "創意與創業" in text_content and ("辦法" in text_content or "pdf" in href.lower()):
                    if href and not href.startswith("javascript"):
                        pdf_url = urljoin(course_base_url, href)
                        saved_path = self.download_pdf(pdf_url, "「創意與創業」學分學程選修辦法.pdf")
                        if saved_path:
                            downloaded_pdfs.append(saved_path)

        except Exception as e:
            logger.error(f"Error crawling Course NCU regulation: {e}", exc_info=True)

        logger.info(f"Course NCU crawl completed. Downloaded {len(downloaded_pdfs)} PDFs.")
        return downloaded_pdfs

    def parse_urls_file(self, urls_file: Path) -> Dict[str, List[Tuple[str, str]]]:
        """Parses urls.txt into markdown targets and special PDF update targets."""
        sections: Dict[str, List[Tuple[str, str]]] = {
            "markdown_pages": [],
            "pdc_regulations": [],
            "course_regulations": [],
        }

        if not urls_file.exists():
            logger.warning(f"URLs file not found: {urls_file}")
            return sections

        with open(urls_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_mode = "markdown"

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if "pdf 檔案更新區" in line.lower() or "pdf 檔案" in line:
                current_mode = "pdf"
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

                # Determine target type
                if "pdc.adm.ncu.edu.tw" in url:
                    sections["pdc_regulations"].append((title, url))
                elif "course.ncu.edu.tw" in url:
                    sections["course_regulations"].append((title, url))
                elif current_mode == "markdown" or "ipeecs.ncu.edu.tw" in url:
                    sections["markdown_pages"].append((title, url))
                else:
                    sections["markdown_pages"].append((title, url))

        return sections

    def crawl_all(self, urls_file: Path) -> Dict[str, Any]:
        """Executes full crawl based on urls.txt."""
        logger.info(f"Parsing URLs config from: {urls_file}")
        targets = self.parse_urls_file(urls_file)

        results: Dict[str, Any] = {
            "markdown_files": [],
            "pdf_files": [],
        }

        # 1. Crawl Web Pages to Markdown
        logger.info(f"Crawling {len(targets['markdown_pages'])} web pages to Markdown...")
        for title, url in targets["markdown_pages"]:
            md_path = self.crawl_markdown_page(title, url)
            if md_path:
                results["markdown_files"].append(md_path)

        # 2. Crawl PDC Regulations
        for title, url in targets["pdc_regulations"]:
            logger.info(f"Crawling PDC Regulation target: {title} ({url})")
            pdfs = self.crawl_pdc_regulations(url)
            results["pdf_files"].extend(pdfs)

        # 3. Crawl Course Regulations
        for title, url in targets["course_regulations"]:
            logger.info(f"Crawling Course NCU target: {title} ({url})")
            pdfs = self.crawl_course_regulation(url)
            results["pdf_files"].extend(pdfs)

        logger.info(
            f"Crawl All Finished. Saved {len(results['markdown_files'])} Markdown files, "
            f"{len(results['pdf_files'])} PDF files."
        )
        return results
