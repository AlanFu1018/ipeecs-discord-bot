# Discord 大學系所智能客服機器人 — 系統規格書與實作計畫

本文件彙整「大學系所專用 Discord 客服機器人」之現行規格、軟體架構設計及實作進度，作為後續開發與維護的依據。操作步驟請參考 [`User_Guide.md`](User_Guide.md)。

---

## 一、 專案概述與核心目標
- **專案名稱**：國立中央大學資電學士班 Discord 智能客服機器人（IPEECS Department Advisor Bot）
- **目標受眾**：資電學士班在校生、轉學生、新生、境外生（僑生／外籍生）及對學系資訊有疑問之人士；以繁體中文為主，並支援以其他語言提問與回答。
- **核心價值**：
  1. 針對系所「修課規定、課程介紹、學分抵免、畢業門檻、校曆、註冊與行政流程」等各類規章提供精準、有依據的即時諮詢。
  2. 透過一對一私訊（DM）提供專屬問答空間，避免伺服器公頻洗版與干擾。
  3. 導入多輪對話改寫與 RAG 檢索，兼顧上下文理解流暢度與低 Token 消耗。
  4. 嚴守客服業務範圍，防範 Prompt Injection 將機器人挪作他用。
  5. 採用高內聚、低耦合的 Adapter 設計模式，方便抽換 LLM 與 Embedding 提供者。

---

## 二、 系統詳細規格需求

### 1. 互動模式與頻道範圍
- **通訊管道**：Discord 一對一私訊（DMChannel）。
- **觸發機制**：自動監聽使用者的每則私訊內容並進行解答；在伺服器頻道中被 `@` 時，回覆引導訊息請使用者改用私訊。
- **會話管理（Session）**：
  - 以使用者 `user_id` 作為獨立 Session 辨識，Session 存於記憶體（重啟即清空）。
  - 保存最近 `max_history_turns`（預設 5）輪問答。
  - 支援閒置自動過期機制（預設 60 分鐘無互動即自動重置對話歷史）。
  - 支援手動重置：整則訊息為 `/reset`、`reset`、`重新開始` 或 `重設` 時清空歷史（文字關鍵字，非 Discord Slash Command）。
- **長訊息處理**：超過 1900 字元的回覆依行拆分為多則訊息發送，以符合 Discord 2000 字元上限。

### 2. 前置分析：語言偵測、問題改寫與範圍判斷
- 每輪對話（含第一則訊息）皆先呼叫一次輕量 Prompt（`temperature=0`），結合對話歷史輸出三項結果：
  - **`LANGUAGE`**：使用者提問所使用的自然語言，記錄於 Session，作為回答語言。
  - **`QUERY`（Query Condensing）**：將代名詞／省略問題（例如：「那大二的必修呢？」）改寫為獨立、語意完整的**繁體中文**檢索問句（如：「113學年度資訊工程專長大二必修課程有哪些？」）。無論提問語言為何，檢索一律以繁體中文進行，以對應中文規章。
  - **`SCOPE`**：`IN`（修課、學分、畢業門檻、學程、轉系／雙主修、規章與行政事項等）或 `OUT`（撰寫程式碼、作業解答、翻譯、一般知識、閒聊等）。
- **失敗降級**：前置分析呼叫失敗時，使用原始問句檢索、沿用前次偵測語言（預設繁體中文），並視為 `IN`，避免 LLM 短暫異常造成全面拒答。
- **優勢**：
  - 向量檢索（RAG）精準度大幅提升。
  - 最終生成回答時僅需帶入「檢索到的相關規章段落 + 使用者原問句 + 改寫後的問句」，不需重複堆疊全部歷史訊息，大幅節省 Token。

### 3. 業務範圍把關與 Prompt Injection 防護
- **隨機標籤隔離**：使用者輸入與對話歷史包在每次隨機生成的 `<data_{hex}>` / `<user_input_{hex}>` 標籤中，並聲明其中之指令、角色設定、身分宣稱、緊急性要求一律無效。
- **程式層拒答**：`SCOPE: OUT` 時不進入檢索與生成，直接回傳標準 Fallback 訊息。
- **生成守則**：回答中的每項規定都必須能對應到某一則 `[參考規章 n]`，否則視為超出範圍；即使使用者宣稱為本系學生或攸關課業也不例外。
- **輸出檢查**：回覆中出現程式碼區塊標記（```` ``` ````）時，攔截並改回傳 Fallback 訊息。
- **記憶隔離**：被拒答的輪次不寫入對話歷史，避免影響後續輪次。

### 4. 雙層 Adapter 抽象架構（解耦設計）
- **LLM Provider 抽象層**（`BaseLLMProvider`）：
  - 現行介接：**Google Gemini API**（`google-genai` SDK，預設模型 `gemini-3.1-flash-lite`），同時用於前置分析、回答生成與表格 PDF 轉 Markdown。
  - 錯誤處理：遇 503（模型過載）以指數退避重試（1→32 秒，最多 6 次）；其他錯誤不重試。
  - 擴充介面：`get_llm_provider()` 工廠函式預留 OpenAI、本地開源模型（Ollama / vLLM）等提供者的擴充位置（尚未實作）。
- **Embedding Provider 抽象層**（`BaseEmbeddingProvider`）：
  - 預設介接：**Google Gemini Embedding API**（`gemini-embedding-001`，3072 維，免本機顯卡資源）。
  - 錯誤處理：批次 10 筆、批次間冷卻 5 秒；遇 429 至少等待 50 秒、其他暫時性錯誤由 15 秒起 ×1.5 退避，最多 8 次。
  - 擴充介面：已實作本地 SentenceTransformer（`embed_local.py`），透過 `embedding.provider: "local"` 切換（需另行安裝 `sentence-transformers`）。

### 5. 知識庫管理與爬蟲同步機制
- **資料來源設定**：`config/urls.yaml`，以 `["標題", "網址"]` 格式分為三區：
  - `web`：資電學士班官網（簡介、公告、招生、辦公室位置與同仁、班務委員）、中央大學官網（願景、簡介、校史、交通、聯絡資訊）、計算機中心（新生帳號、Google Workspace）。
  - `text_pdf`：中央大學學則（PDF）、資工系會議室教室教學實驗室管理細則（DOCX）。
  - `table_pdf`：教務章則彙編中各學年度之資電學士班及四專長課規、「創意與創業」學分學程選修辦法、最新學年度校曆。
- **資料分區（zone）**：
  - **`data`（資料區）**：由爬蟲自動維護，每次同步可能被覆蓋。
  - **`fixed`（固定資料區）**：`res/data/fixed/markdown/`，人工整理並納入版本控制（目前含新生註冊流程、第二學期註冊、居留證延期、僑生保險、在學／工作證明、中英版學生宿舍管理辦法），同步時直接解析，不經爬蟲。
  - 每個 Chunk 的 metadata 皆標記 `zone`，可單獨清空、重建某一區。
- **同步腳本（`sync_data.py`）**：
  - 獨立於機器人主程式，手動或透過排程執行；支援 `--zone`、`--skip-crawl`、`--skip-llm-convert`、`--skip-converted`、`--no-reset` 參數。
  - 網頁正文轉為 Markdown 儲存於 `res/data/markdown/`。
  - 文字型 PDF 以 `pymupdf4llm` 解析；DOCX 直接解析段落與表格。
  - 表格型 PDF 交由 Gemini 多模態轉為 Markdown 表格後快取於 `res/data/markdown/`（429/503 時遞增等待重試，最多 5 次）。
- **分塊（Chunking）規格**：
  - 依 Markdown 標題將含表格段落與純文字段落分離，**表格段落整段保留為單一 Chunk**，不被切斷。
  - 純文字段落以 800 字元滑動視窗切分、重疊 130 字元，並在視窗後半段貼齊換行或標點。
  - 過濾長度不超過 100 字元（`chunk_mini`）的碎片。
  - 每個 Chunk 開頭注入來源文件標題，保留學年度／專長等上下文。
- **向量資料庫**：
  - 使用本地持久化 **ChromaDB**（Cosine Distance），向量索引存於 `res/data/chroma_db/`，機器人啟動時直接讀取，平時運行不重複計算檔案 Embedding。
  - 檢索取前 `top_k`（預設 9）個片段。

### 6. 回答規範與例外策略
- **回答語言**：使用前置分析偵測到的語言回答；參考規章為中文時翻譯後呈現。
- **格式規範**：禁止輸出 Markdown 表格，改以條列呈現；嚴格區分「院訂必修」與「院訂必選」。
- **主動追問**：問題缺少入學學年度、專長領域、年級或身分等關鍵條件時，先提供概括說明，再主動追問細節。
- **來源標註**：依據檢索資料回答時，文末附註來源（例如：`📌 參考來源：113學年度_資訊工程專長.md`）；能正常回答時**不**附帶系辦聯絡資訊。
- **例外與降級處理**：查無資料、超出範圍、生成失敗或輸出被攔截時，嚴格禁止模型臆測，統一回覆標準格式（聯絡資訊取自 `config.yaml` 的 `department_info`）：
  > 我目前在規章資料庫中查無足夠的相關資訊（問題超出規章範圍或查無記載）。
  >
  > 若我有無法回答的問題，或是需要進一步協助，也歡迎透過以下方式聯繫系辦公室：
  > 🏢 資訊電機學院學士班辦公室
  > 📞 電話：03-4227151 分機 35007
  > 📧 信箱：ncu35007@ncu.edu.tw
  > 📍 位置：工程五館E6 B棟106室 (E6-B106)
  > ⏰ 服務時間：週一至週五 08:30 - 17:00

  系統異常時開頭改為「抱歉，目前系統處理時發生異常。」；Discord 層發生未預期錯誤時回覆「抱歉，處理您的訊息時發生錯誤，請稍後再試。」

---

## 三、 專案目錄結構

```text
ipeecs-discord-bot/
├── config/
│   ├── config.yaml          # 系統參數設定（模型名稱、Top-k、分塊、過期時間、路徑、系辦預設資訊等）
│   └── urls.yaml            # 目標爬取網址清單（web / text_pdf / table_pdf 三區）
├── res/
│   └── data/
│       ├── raw/             # 下載的原始文件 (需加入 .gitignore)
│       │   ├── text_pdfs/   #   文字為主的 PDF / DOCX
│       │   └── table_pdfs/  #   表格為主的 PDF（課規、學程、校曆）
│       ├── markdown/        # 網頁爬蟲與 Gemini 表格轉換後的 Markdown (需加入 .gitignore)
│       ├── fixed/
│       │   └── markdown/    # 固定資料區：人工維護的 Markdown（納入版本控制）
│       └── chroma_db/       # ChromaDB 本地向量資料庫 (需加入 .gitignore)
├── doc/
│   ├── Discord_Bot_Spec.md            # 初版規格書（保留作歷史紀錄）
│   ├── Discord_Bot_Spec_Updated.md    # 現行規格書（本文件）
│   └── User_Guide.md                  # 使用與維運手冊
├── src/ipeecs_bot/
│   ├── core/                # 核心基礎設施
│   │   ├── config.py        # 集中讀取 YAML 與 .env 的設定載入器
│   │   └── logger.py        # 統一終端機日誌輸出格式
│   ├── llm_api/             # LLM 與 Embedding 抽象介面
│   │   ├── __init__.py      # get_llm_provider / get_embedding_provider 工廠函式
│   │   ├── llm_base.py      # LLM 抽象基底介面
│   │   ├── llm_model/
│   │   │   └── llm_gemini.py    # Gemini LLM（含 503 指數退避重試）
│   │   ├── embed_base.py    # Embedding 抽象基底介面
│   │   └── embed_method/
│   │       ├── embed_gemini.py  # Gemini Embedding（批次、速率控制、重試）
│   │       └── embed_local.py   # 本地 SentenceTransformer
│   ├── rag/                 # RAG 知識庫與向量管理
│   │   ├── parser.py        # Markdown / PDF / DOCX 解析與表格感知切塊
│   │   └── vector_store.py  # ChromaDB 向量寫入、分區清除與檢索
│   ├── services/            # 業務邏輯與狀態管理
│   │   ├── crawler_main.py  # DataCrawler：三分區爬取調度、下載、Gemini 表格轉 Markdown
│   │   ├── crawlers/
│   │   │   ├── web_crawler.py    # 網頁正文轉 Markdown
│   │   │   ├── text_crawler.py   # 學則 PDF、資工系管理細則 DOCX
│   │   │   └── table_crawler.py  # 各學年度課規、學分學程、最新校曆
│   │   ├── session.py       # 對話歷史、語言記錄與 Session 過期管理
│   │   └── chat_service.py  # 前置分析、範圍把關、RAG 檢索與回覆生成
│   └── bot/
│       └── bot.py           # Discord Bot 主程式（DM 處理、訊息分段）
├── tests/                   # 測試與實驗腳本（重試機制、Session、Parser 等）
├── .env.example             # 環境變數範本
├── .env                     # 本地真實環境變數 (需加入 .gitignore)
├── requirements.txt         # Python 依賴套件清單
├── sync_data.py             # 獨立資料同步工具（爬蟲、解析與建庫）
└── main.py                  # 機器人啟動入口
```

---

## 四、 核心資料流與流程圖

### 1. 問答流程
```text
[使用者發送 Discord DM]
           │
           ▼
[bot.py] 接收私訊（公頻被 @ 時僅回覆引導訊息）並轉交 chat_service.py
           │
           ▼
[session.py] 取得 Session（閒置逾 60 分鐘則重置）
           ├── 訊息為重置關鍵字 ──> 清空歷史並回覆確認訊息
           ▼
[chat_service.py → llm_gemini.py] 前置分析：LANGUAGE / QUERY（繁中獨立問句）/ SCOPE
           ├── SCOPE = OUT ──> 回傳 Fallback 訊息（不寫入歷史）
           ▼
[vector_store.py] 以 embed_gemini.py 向量化問句，於 ChromaDB 檢索 Top-9 相關段落
           ├── 無結果 ──> Fallback 訊息
           ▼
[chat_service.py] 組合 System Prompt（語言、回答守則、規章段落）+ 隨機標籤包覆的使用者問題
           │
           ▼
[llm_gemini.py] 生成解答（503 時自動重試）
           ├── 生成失敗 ──> 系統異常 Fallback 訊息
           ├── 輸出含程式碼區塊 ──> Fallback 訊息（不寫入歷史）
           ▼
[session.py] 寫入本輪問答
           │
           ▼
[bot.py] 依 1900 字元拆分後發送回使用者 DM 視窗
```

### 2. 資料同步流程（`sync_data.py`）
```text
[Step 1] 爬取與轉換（zone=fixed 或 --skip-crawl 時略過）
   urls.yaml ─┬─ web       ──> web_crawler   ──> res/data/markdown/*.md
              ├─ text_pdf  ──> text_crawler  ──> res/data/raw/text_pdfs/*.pdf|.docx
              └─ table_pdf ──> table_crawler ──> res/data/raw/table_pdfs/*.pdf
                                                   └─ Gemini 轉換 ──> res/data/markdown/*.md
                                                      （--skip-llm-convert / --skip-converted 控制）
[Step 2] 解析與分塊
   data  區：res/data/markdown + res/data/raw/text_pdfs  ──> Chunks (zone=data)
   fixed 區：res/data/fixed/markdown                     ──> Chunks (zone=fixed)
[Step 3] 向量化與寫入
   清空（zone=all 清全庫；指定 zone 只清該區；--no-reset 不清）
   ──> Gemini Embedding（批次 + 冷卻 + 重試）──> ChromaDB
```

---

## 五、 階段性實行計畫與進度（Roadmap）

### 【階段一：環境配置與金鑰準備】 ✅ 已完成
1. 建立 Python 3.10+ 虛擬環境並安裝核心相依套件（`discord.py`, `google-genai`, `chromadb`, `pymupdf4llm`, `beautifulsoup4`, `requests`, `pyyaml`, `python-dotenv`）。
2. 在 Discord Developer Portal 建立 Bot，開啟 **Message Content Intent**，並加入伺服器。
3. 取得 Google Gemini API Key，以 `.env.example` 為範本建立 `.env`。

### 【階段二：資料同步與 RAG 向量庫建立（`sync_data.py`）】 ✅ 已完成
1. 實作 `services/crawler_main.py` 與 `services/crawlers/`：讀取 `urls.yaml`，網頁正文轉 Markdown，文字／表格 PDF（及 DOCX）分區下載。
2. 以 Gemini 多模態將表格 PDF 轉換為 Markdown，解決跨頁表格截斷問題，並支援 `--skip-llm-convert`、`--skip-converted` 節省 Token。
3. 實作 `rag/parser.py`：文字清洗、表格感知切塊、標題注入、雜訊過濾。
4. 實作 `embed_gemini.py` 與 `vector_store.py`：批次向量化、速率控制並寫入本地 ChromaDB。
5. 新增固定資料區與 `--zone` 參數，支援人工資料與爬蟲資料分開維護。

### 【階段三：Adapter 抽換層與對話核心邏輯（`llm_api/` & `services/`）】 ✅ 已完成
1. 定義 `BaseLLMProvider` 與 `BaseEmbeddingProvider` 抽象介面，實作 `GeminiLLMProvider`、`GeminiEmbeddingProvider`、`LocalEmbeddingProvider`。
2. 實作 `services/session.py`：`user_id -> UserSession`，滑動視窗、過期清理與語言記錄。
3. 實作 `services/chat_service.py`：
   - **前置分析模組**：語言偵測 + 多輪改寫 + 範圍判斷。
   - **檢索模組**：向 ChromaDB 查詢最相關之規章段落。
   - **生成模組**：建構 Prompt 指引模型以使用者語言精確回答並附加來源；若無匹配段落或超出範圍則觸發系辦聯絡資訊 fallback。

### 【階段四：Discord 機器人主體開發（`bot.py` & `main.py`）】 ✅ 已完成
1. 實作 `bot/bot.py`：
   - 監聽 `on_ready` 輸出上線資訊並設定狀態。
   - 監聽 `on_message`：限定處理私訊，排除機器人自身發言；公頻被 `@` 時引導至私訊。
   - 整合 `typing()` 狀態提示，調用 `chat_service.py` 並分段回應。
   - 以文字關鍵字（`/reset` 等）清空對話狀態。
2. 撰寫 `main.py` 作為系統啟動入口，含設定檢查、空資料庫警告、優雅關閉與日誌機制。

### 【階段五：系統測試、情境驗證與調整】 ✅ 已完成（持續進行）
1. **單輪基準測試**：基本修課規章、學分查詢，確認來源引用與回答準確度；調整 chunk 大小與 `top_k` 使回答更完整。
2. **多輪指代測試**：「那大三呢？」、「如果抵免的話門檻是什麼？」等追問，驗證改寫機制。
3. **邊界與例外測試**：詢問非系所問題、要求撰寫程式、偽稱身分或急迫性等，驗證範圍把關與 Prompt Injection 防護。
4. **重試機制測試**：以 mock 驗證 Embedding 429 與 LLM 503 重試行為（`tests/test_embed_retry.py`、`tests/test_llm_retry.py`），不實際呼叫 API。

### 【階段六：功能強化】 ✅ 已完成
1. 多語言提問與回答。
2. 最新學年度校曆、資工系管理細則（DOCX）、中央大學與計算機中心網頁等資料來源擴充。
3. 固定資料區納入境外生相關行政資料（註冊、居留證、保險、宿舍等）。
4. LLM 503 指數退避重試。

### 【未來規劃】
1. 實作 OpenAI 或本地 LLM（Ollama / vLLM）Provider。
2. Session 持久化（目前重啟後對話記憶即清空）。
3. 將 `/reset` 註冊為正式 Discord Slash Command。
4. 同步流程排程化，定期自動更新規章與校曆。
