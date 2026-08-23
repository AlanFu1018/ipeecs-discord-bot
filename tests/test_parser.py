import pymupdf4llm

import src.ipeecs_bot.rag.parser as parser

"""Parses a PDF file using pymupdf4llm with table cleaning and returns chunks."""

file_path = "C:/ALL FILES/Code/ipeecs_discord_bot/ipeecs-discord-bot/res/data/raw/「創意與創業」學分學程選修辦法.pdf"
raw_md = pymupdf4llm.to_markdown(str(file_path))
p = parser.DocumentParser(
    chunk_size=1500,
    chunk_overlap=200,
)
cleaned_md = p.clean_pdf_markdown(raw_md)
print(cleaned_md)
w = input("輸入")
# title = file_path.stem
# metadata = {
#     "source": file_path.name,
#     "file_path": str(file_path),
#     "title": title,
#     "doc_type": "pdf",
# }