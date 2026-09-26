from src.ipeecs_bot.rag import vector_store
from src.ipeecs_bot.llm_api import embed_base, get_embedding_provider
from src.ipeecs_bot.core import config

settings = config.get_settings()
embedding_provider = get_embedding_provider(settings)
vector_store = vector_store.VectorStore(
    persist_dir=settings.chroma_db_dir,
    collection_name=settings.collection_name,
    embedding_provider=embedding_provider,
)

results = vector_store.search_sync("我是110的學生需要修CO2001嗎", settings.top_k)

n = 1
for i in results:
    print(f"-------------------- result {n} --------------------")
    print(i["metadata"]["title"])
    print(i["content"])
    print("")
    n += n