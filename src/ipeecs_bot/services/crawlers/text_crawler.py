from pathlib import Path
from typing import List
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from src.ipeecs_bot.core.logger import logger

class TextCrawlerMixin:
    """Methods for crawling to get text-based pdfs"""

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