from bs4 import BeautifulSoup
from typing import Optional
from pathlib import Path
import re

from src.ipeecs_bot.core.logger import logger

class WebCrawlerMixin:
    """Methods for crawling website, downloading it into .md"""

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