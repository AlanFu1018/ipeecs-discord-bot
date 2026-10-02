from pathlib import Path
from src.ipeecs_bot.core import config
# from new_parser import DocumentParser, DocumentChunk
from src.ipeecs_bot.rag.parser import DocumentParser, DocumentChunk
settings = config.get_settings()
file_path = settings.markdown_dir / "114學年度_資訊電機學院學士班【1140924修訂版】.md"
parser = DocumentParser(chunk_size=800, chunk_overlap=130)

chunks = parser.parse_pdf_file(file_path)
count = 1
for i in chunks:
    print(f"chunk {count} : \n {i.content}")
    count += 1
print("end")
