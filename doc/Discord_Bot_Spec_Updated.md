# Discord 大學系所智能客服機器人 — 系統規格書與實作計畫

本文件彙整「大學系所專用 Discord 客服機器人」之完整定案規格、軟體架構設計及階段性實行計畫，作為後續開發與維護的依據。

---

## 一、 專案概述與核心目標
- **專案名稱**：大學系所 Discord 智能客服機器人（Department FAQ & Advisor Bot）
- **目標受眾**：繁體中文（台灣）使用者、在校生、轉學生、新生及對學系資訊有疑問之人士。
- **核心價值**：
  1. 針對系所「修課規定、課程介紹、學分抵免、畢業門檻」等各類規章提供精準、有依據的即時諮詢。
  2. 透過一對一私訊（DM）提供專屬問答空間，避免伺服器公頻洗版與干擾。
  3. 導入多輪對話改寫與 RAG 檢索，兼顧上下文理解流暢度與低 Token 消耗。
  4. 採用高內聚、低耦合的 Adapter 設計模式，方便隨時抽換 LLM 與 Embedding 提供者。

---

## 二、 系統詳細規格需求

### 1. 互動模式與頻道範圍
- **通訊管道**：Discord 一對一私訊（DMChannel）。
- **觸發機制**：自動監聽使用者的每則私訊內容並進行解答。
- **會話管理（Session）**：
  - 以使用者 `user_id` 作為獨立 Session 辨識。
  - 支援閒置自動過期機制（預設 15 分鐘無互動即自動重置對話歷史）。
  - 支援手動重置指令（如 `/reset` 或輸入「重新開始」清空歷史）。

### 2. 對話記憶與 Token 最佳化（獨立問題改寫法）
- **改寫機制（Query Condensing）**：
  - 當使用者進行多輪對話時（例如：「那大二的必修呢？」、「如果抵免的話怎麼算？」），系統先呼叫輕量 Prompt，結合前 2~3 輪歷史對話，將使用者代名詞問題重寫為獨立且語意完整的問句（如：「資訊工程學系大二必修課程有哪些？」）。
- **優勢**：
  - 向量檢索（RAG）精準度大幅提升。
  - 最終生成回答時僅需帶入「檢索到的相關規章段落 + 改寫後的問句」，不需重複堆疊全部歷史訊息，大幅節省 Token。

### 3. 雙層 Adapter 抽象架構（解耦設計）
- **LLM Provider 抽象層**：
  - 預設介接：**Google Gemini API**（Gemini 1.5 / 2.0 Flash，免費額度充足且繁中表現佳）。
  - 擴充介面：預留 OpenAI、本地開源模型（Ollama / vLLM）介面，未來可無痛切換。
- **Embedding Provider 抽象層**：
  - 預設介接：**Google Gemini Embedding API**（text-embedding-004，免本機顯卡資源）。
  - 擴充介面：預留本地 SentenceTransformer（如 paraphrase-multilingual / bge-m3）介面，透過設定檔一鍵切換。

### 4. 知識庫管理與爬蟲同步機制
- **資料來源**：
  - `config/urls.txt`：列出系所重要靜態網頁（如修課規定、師資介紹、畢業標準）。
  - 關聯 PDF 文件：爬蟲自動下載網頁中連結的規章 PDF，或由管理者手動放置至 `res/data/raw/`。
- **同步腳本（`sync_data.py`）**：
  - 獨立於機器人主程式，手動或透過排程執行。
  - 網頁轉為 Markdown 儲存於 `res/data/markdown/`。
  - 解析 PDF 與 Markdown，進行文字清洗與分塊（Chunking，預設 500~800 字元，重疊 100 字元）。
- **向量資料庫**：
  - 使用本地持久化 **ChromaDB**，向量索引直接存於本機硬碟，機器人啟動時直接讀取，平時運行不重複計算檔案 Embedding。

### 5. 回答規範與例外策略
- **來源標註**：回答內容若依據檢索到的資料，文末必須附註來源檔案名稱或網址（例如：`

📌 參考來源：113學年度學士班修業規章.pdf`）。
- **例外與降級處理**：若資料庫內查無足夠資訊，嚴格禁止模型胡言亂語，統一回覆標準格式：
  > 「我目前沒有這方面的資訊，建議直接聯繫系辦公室洽詢。
📞 系辦電話：[預設電話]
📧 系辦信箱：[預設信箱]
📍 系辦位置：[預設辦公室位置]」

---

## 三、 專案目錄結構

```text
ipeecs-discord-bot/
├── config/
│   ├── config.yaml          # 系統參數設定（模型名稱、Top-k、過期時間、系辦預設資訊等）
│   └── urls.txt             # 目標爬取之系網網址清單
├── res/
│   └── data/    
│       ├── raw/             # 存放下載的 PDF 與原始文件
│       ├── markdown/        # 網頁爬蟲萃取轉換後的 Markdown 檔案
│       └── chroma_db/       # ChromaDB 本地向量資料庫儲存目錄 (需加入 .gitignore)
├── doc/
│   └── Discord_Bot_Spec.md            # 專案說明與運行指引
├── src/ipeecs_bot/
│   ├── core/                # 核心基礎設施
│   │   ├── __init__.py
│   │   ├── config.py        # 集中讀取 YAML 與 .env 的設定載入器
│   │   └── logger.py        # 統一終端機日誌輸出格式 
│   ├── llm_api/             # LLM 與 Embedding 抽象介面
│   │   ├── __init__.py
│   │   ├── llm_base.py      # LLM 抽象基底介面
│   │   ├── llm_model/
│   │   │   ├── __init__.py
│   │   │   └── llm_gemini.py
│   │   ├── embed_base.py    # Embedding 抽象基底介面
│   │   └── embed_method/
│   │       ├── __init__.py
│   │       ├── embed_gemini.py
│   │       └── embed_local.py
│   ├── rag/                 # RAG 知識庫與向量管理
│   │   ├── __init__.py
│   │   ├── parser.py        # PDF 與 Markdown 文本切塊器
│   │   └── vector_store.py  # ChromaDB 向量檢索器
│   ├── services/            # 業務邏輯與狀態管理
│   │   ├── __init__.py
│   │   ├── crawler.py       # 網頁爬蟲與 PDF 下載器
│   │   ├── session.py       # 對話歷史與 Session 過期管理
│   │   └── chat_service.py  # 整合問題改寫、RAG 檢索與回覆生成
│   └── bot/
│       ├── __init__.py
│       └── bot.py           # Discord Bot 主程式
├── tests/                   # 獨立測試模組
├── .env                     # 本地真實環境變數 
├── requirements.txt         # Python 依賴套件清單
├── sync_data.py             # 獨立資料同步工具（執行爬蟲、解析與建庫）
└── main.py                  # 機器人啟動入口
```

---

## 四、 核心資料流與流程圖

```text
[使用者發送 Discord DM]
           │
           ▼
[bot.py] 接收訊息並轉交給 chat_service.py
           │
           ▼
[session.py] 檢查該使用者是否有未過期之對話歷史
           ├── 有歷史 ──> [llm_gemini.py] 將問題改寫為「獨立完整問句」
           └── 無歷史 ──> 直接使用使用者當前問句
           │
           ▼
[vector_store.py] 使用 embed_gemini.py 向量化問句，並於 ChromaDB 檢索 Top-3 相關段落
           │
           ▼
[chat_service.py] 組合 System Prompt + 規章段落 + 獨立問句
           │
           ▼
[llm_gemini.py] 生成繁體中文解答（附註參考來源 / 查無資料時提供系辦聯絡方式）
           │
           ▼
[bot.py] 將回覆發送回使用者 DM 視窗，並更新 session.py 對話紀錄
```

---

## 五、 階段性實行計畫（Roadmap）

### 【階段一：環境配置與金鑰準備】
1. **建立專案環境**：建立 Python 3.10+ 虛擬環境（venv）並安裝核心相依套件（`discord.py`, `google-genai`, `chromadb`, `pypdf`, `beautifulsoup4`, `requests`, `pyyaml`, `python-dotenv`）。
2. **申請 Discord 應用程式與憑證**：
   - 在 Discord Developer Portal 建立 Bot。
   - 開啟 Privileged Gateway Intents 中的 **Message Content Intent**。
   - 產出邀請連結並將 Bot 加入測試伺服器。
3. **申請 Google Gemini API Key**：取得 API Key 並建立 `.env` 設定檔。

### 【階段二：資料同步與 RAG 向量庫建立（`sync_data.py`）】
1. 實作 `src/ipeecs_bot/services/crawler.py`：讀取 `urls.txt`，萃取主要正文轉換為 Markdown，並過濾出 PDF 檔案下載至 `res/data/raw/`。
2. 實作 `src/ipeecs_bot/rag/parser.py`：清洗文字、去除多餘空行，將長文本切分為 500~800 字元的語意段落（Chunk），保留檔案來源 metadata。
3. 實作 `src/ipeecs_bot/llm_api/embed_method/embed_gemini.py` 與 `src/ipeecs_bot/rag/vector_store.py`：將文本塊批次轉換為向量，寫入本地 ChromaDB。
4. 撰寫 `sync_data.py` 整合以上流程，達成「一鍵同步與建庫」。

### 【階段三：Adapter 抽換層與對話核心邏輯（`llm_api/` & `services/`）】
1. 定義 `BaseLLMProvider` 與 `BaseEmbeddingProvider` 抽象介面，實作 `GeminiLLMProvider` 與 `GeminiEmbeddingProvider`。
2. 實作 `src/ipeecs_bot/services/session.py`：字典維護 `user_id -> List[Message]`，設定滑動視窗上限與過期清理機制。
3. 實作 `src/ipeecs_bot/services/chat_service.py`：
   - **改寫模組**：歷史多輪對話改寫為單一完整問句。
   - **檢索模組**：向 ChromaDB 查詢最相關之規章段落。
   - **生成模組**：建構 Prompt 指引模型精確回答並附加來源標籤；若無匹配段落則觸發系辦聯絡資訊 fallback。

### 【階段四：Discord 機器人主體開發（`bot.py` & `main.py`）】
1. 實作 `src/ipeecs_bot/bot/bot.py`：
   - 監聽 `on_ready` 輸出上線資訊。
   - 監聽 `on_message`：限定處理私訊（`isinstance(channel, discord.DMChannel)`），排除機器人自身發言。
   - 整合 `typing()` 狀態提示，調用 `chat_service.py` 並回應用戶。
   - 實作 `/reset` 指令供手動清空對話狀態。
2. 撰寫 `main.py` 作為系統啟動入口，並加入優雅關閉（Graceful Shutdown）與日誌（Logging）機制。

### 【階段五：系統測試、情境驗證與調整】
1. **單輪基準測試**：測試基本修課規章、學分查詢，確認來源引用與回答準確度。
2. **多輪指代測試**：測試如「那大三呢？」、「如果抵免的話門檻是什麼？」等代名詞追問，驗證改寫機制。
3. **邊界與例外測試**：詢問非系所問題（如天氣、娛樂），驗證是否能嚴格回傳系辦公室聯絡方式。
4. **模型切換驗證**：測試透過設定檔抽換 Embedding 或 LLM Adapter 的可行性。
