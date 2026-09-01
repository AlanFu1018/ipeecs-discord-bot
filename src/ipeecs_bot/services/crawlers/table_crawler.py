import re
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from src.ipeecs_bot.core.logger import logger

class TableCrawlerMixin:
    """Methods for crawling to get table-based pdfs"""

    def crawl_latest_academic_calendar(self, calendar_page_url: str) -> List[Path]:
        """Crawls NCU Academic Calendar page (pdc 1725) and downloads the latest academic year calendar PDF into table_pdf_dir."""
        logger.info(f"Crawling latest NCU academic calendar from: {calendar_page_url}")
        downloaded_pdfs: List[Path] = []

        try:
            resp = self.fetch_url(calendar_page_url)
            soup = BeautifulSoup(resp.text, "html.parser")

            candidates: List[Tuple[int, str, str]] = []

            for a in soup.find_all("a"):
                title_attr = a.get("title", "")
                text_content = a.get_text(strip=True)
                href = a.get("href", "")
                combined = f"{text_content} {title_attr}"

                if not href or href.startswith("javascript"):
                    continue

                # Match Chinese academic calendar links, e.g. "開啟 115 學年度校曆" or "115 學年度校曆" or "calendar115.pdf"
                year_match = re.search(r"(\d{3})\s*學年度.*校曆", combined) or re.search(r"calendar(\d{3})\.pdf", href.lower())
                if year_match:
                    year = int(year_match.group(1))
                    pdf_url = urljoin(calendar_page_url, href)
                    candidates.append((year, pdf_url, f"{year}學年度校曆"))

            if candidates:
                # Sort descending by academic year to pick the newest/latest
                candidates.sort(key=lambda x: x[0], reverse=True)
                latest_year, latest_url, file_label = candidates[0]
                logger.info(f"Found latest academic calendar: {latest_year}學年度 -> {latest_url}")

                saved_path = self.download_pdf(
                    latest_url,
                    f"{file_label}.pdf",
                    target_dir=self.table_pdf_dir,
                )
                if saved_path:
                    downloaded_pdfs.append(saved_path)
            else:
                logger.warning(f"No academic calendar PDF links found on: {calendar_page_url}")

        except Exception as e:
            logger.error(f"Error crawling academic calendar: {e}", exc_info=True)

        return downloaded_pdfs

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

                            combined_item_norm = re.sub(r"\s+", "", combined_item)
                            for kw in target_keywords:
                                if kw in combined_item_norm:
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