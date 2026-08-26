"""Crawlers module for all kinds of crawlers we need for get our newest info from web."""
from .table_crawler import TableCrawlerMixin
from .text_crawler import TextCrawlerMixin
from .web_crawler import WebCrawlerMixin

__all__ = [
    "TableCrawlerMixin",
    "TextCrawlerMixin",
    "WebCrawlerMixin"
]