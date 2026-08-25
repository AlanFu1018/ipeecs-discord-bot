# 國立中央大學資電學士班智能客服機器人 — 使用手冊

歡迎使用 **國立中央大學資訊電機學院學士班（IPEECS）智能客服機器人**。本手冊旨在提供使用者操作與維運指引，及常見問題排解。

---

## 📖 目錄
1. [系統概述與核心功能](#1-系統概述與核心功能)
2. [環境需求與前置準備](#3-環境需求與前置準備)
3. [安裝與設定指南](#4-安裝與設定指南)
4. [知識庫資料同步 (sync_data.py)](#5-知識庫資料同步-sync_datapy)
5. [啟動與運行機器人 (main.py)](#6-啟動與運行機器人-mainpy)
6. [Discord 互動與提問指引](#7-discord-互動與提問指引)
7. [系統維護與客製化](#9-系統維護與客製化)

---

## 1. 系統概述與核心功能

- **一對一私訊諮詢 (DM)**：所有對話皆在私訊中進行，保護學生修課隱私且不干擾公開伺服器。
- **高精確度規章檢索**：自動爬取並索引 109～114 學年度各專長課規（資工、電機、通訊、網工）、中央大學學則、創意與創業學分學程及系網資訊，回答均標註來源。
- **Gemini 多模態表格轉 Markdown 技術**：針對大量複雜表格的修業規章 PDF，透過 Gemini 多模態解析轉換為乾淨 Markdown 表格再分塊索引，大幅提升表格檢索精確度。
- **多輪對話改寫 (Query Condensing)**：具備代名詞與省略語還原能力（例如接續問「那大三呢？」、「抵免門檻是多少？」），自動改寫為獨立語意問句後進行精準向量檢索。
- **主動追問與引導**：當使用者問題較為籠統（如未指明入學學年度或專長領域）時，機器人會先給予概述並親切追問細節以提供最精確答案。
- **嚴謹的 Fallback 機制**：遇查無資料、超出範圍或系統異常時，嚴禁模型幻覺，並統一附上資電學士班系辦公室的聯絡資訊。
- **訊息長度自動分段**：針對超過 Discord 2000 字元上限的長回覆，自動分段循序發送，確保資訊完整不中斷。

---

## 2. 環境需求與前置準備

### 系統需求
- **作業系統**：Windows / Linux / macOS
- **Python 版本**：Python 3.10 以上
- **記憶體**：建議 2GB 以上

### 申請金鑰與憑證
1. **Discord Bot Token**：
    - 前往 [Discord Developer Portal](https://discord.com/developers/applications) 建立 Application。
    - 進入 **Bot** 分頁，建立 Bot 並複製 Token。
    - **重要**：在 **Privileged Gateway Intents** 區塊中，開啟 **`MESSAGE CONTENT INTENT`**。
    - 在 **OAuth2 -> URL Generator** 中勾選 `bot` 及相應權限（如 Send Messages、Read Message History），生成邀請連結將機器人加入伺服器。
2. **Google Gemini API Key**：
    - 前往 [Google AI Studio](https://aistudio.google.com/) 申請 API Key。

---

## 3. 安裝與設定指南

### 步驟 1：建立虛擬環境並安裝套件
在專案根目錄開啟終端機（Terminal）：

```bash
# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境 (Windows PowerShell)
.venv\Scripts\activate
# 啟動虛擬環境 (Linux / macOS)
source .venv/bin/activate

# 安裝相依套件
pip install -r requirements.txt
```

### 步驟 2：設定環境變數 (`.env`)
在專案根目錄建立或編輯 `.env` 檔案，填入相應金鑰：

```env
DISCORD_BOT_TOKEN="你的_DISCORD_BOT_TOKEN"
GEMINI_API_KEY="你的_GEMINI_API_KEY"
```

### 步驟 3：調整參數設定 (`config/config.yaml`)
`config/config.yaml` 可調整系統運作參數：

```yaml
bot:
  command_prefix: "!"
  session_timeout_minutes: 60     # 對話閒置過期時間 (分鐘)
  max_history_turns: 5           # 記憶對話輪數上限

llm:
  provider: "gemini"             # gemini / local
  model: "gemini-3.1-flash-lite" # LLM 模型名稱
  temperature: 0.2
  max_output_tokens: 1500

embedding:
  provider: "gemini"             # gemini / local
  model: "gemini-embedding-001"  # Embedding 模型名稱
  dimension: 3072
  batch_size: 10                 # 每批次向量化文件數量 (防 429 頻率限制)
  delay_seconds: 5.0             # 批次之間的冷卻間隔時間 (秒)

rag:
  top_k: 9                       # 檢索前 K 個相關片段
  chunk_size: 600                # 文件分塊字元數
  chunk_overlap: 100             # 分塊重疊字元數
  collection_name: "ipeecs_knowledge_base"

paths:
  urls_file: "config/urls.txt"
  raw_dir: "res/data/raw"
  markdown_dir: "res/data/markdown"
  chroma_db_dir: "res/data/chroma_db"

department_info:
  name: "資訊電機學院學士班辦公室"
  phone: "03-4227151 分機 35007"
  email: "ncu35007@ncu.edu.tw"
  location: "工程五館E6 B棟106室 (E6-B106)"
  office_hours: "週一至週五 08:30 - 17:00"
```

---

## 4. 知識庫資料同步 (sync_data.py)

當系所規章有更新、或初次部署專案時，需執行 `sync_data.py` 建立／更新向量資料庫，並重啟機器人。

### 三分區爬取與轉換架構
資料爬蟲將 `config/urls.txt` 中的目標分為三個分區處理：
1. **網站分區（Zone 1: 網頁）**：爬取系所網頁內容並轉為 Markdown 儲存至 `res/data/markdown`。可以直接將新連結加入，會自動爬取該網站。
2. **文字為主分區（Zone 2: 文字 PDF）**：下載中央大學學則等長文規章至 `res/data/raw/text_pdfs`，以 `pymupdf4llm` 高速解析。若有新連結加入，需要修改網頁爬蟲程式定位 pdf 下載位置。
3. **表格為主分區（Zone 3: 表格 PDF）**：下載 109～114 學年度各專長課規及學程選修辦法 PDF 至 `res/data/raw/table_pdfs`，調用 Gemini 多模態精準轉換為結構化 Markdown 表格並快取。若有新連結加入，需要修改網頁爬蟲程式定位 pdf 下載位置。

```bash
# 完整同步：爬取系網頁面、下載最新規章 PDF、表格轉換並寫入向量庫
python sync_data.py
```

### 進階參數說明

| 參數 | 說明 | 適用情境 |
| :--- | :--- | :--- |
| *(無參數)* | 執行完整爬蟲 + Gemini 表格轉 Markdown + 解析分塊 + 清除舊庫並重新建立索引。 | 定期規章大改版或初次建庫。 |
| `--skip-crawl` | **略過網路爬蟲與表格轉換**，僅重新解析本機 `res/data` 現有文件並快速重建向量索引。 | 已手動加入新 PDF 或修訂 Markdown 時。 |
| `--skip-llm-convert` | **執行爬蟲與下載，但略過 Gemini 表格轉 Markdown**，直接沿用既有 Markdown 快取。 | 更新網頁或下載新 PDF，但不想重複消耗 LLM Token 重新轉表時。 |
| `--skip-converted` | **跳過已轉換過的表格 PDF**，若 Markdown 目錄中已存在同名檔案則略過該 PDF 的 Gemini 轉換。 | 爬取或新增 PDF 時，僅針對尚未轉換的檔案呼叫 LLM 轉表，節省 Token 與時間。 |
| `--no-reset` | **不清空現有資料庫**，直接將新分塊追加（Upsert）進 ChromaDB。 | 單純擴充資料，保留原有向量時。 |

範例：
```bash
# 僅重新解析本機現有文件並建立向量庫（極速）
python sync_data.py --skip-crawl

# 重新爬取網頁與下載 PDF，但略過 Gemini 轉表（節省 Token）
python sync_data.py --skip-llm-convert

# 爬取並僅對尚未轉成 Markdown 的新 table_pdfs 進行 Gemini 轉換
python sync_data.py --skip-converted
```

---

## 5. 啟動與運行機器人 (main.py)

資料庫同步完成後，即可啟動 Discord 機器人：

```bash
python main.py
```

終端機輸出範例：
```text
[INFO] Initializing IPEECS Discord Bot services...
[INFO] Initialized VectorStore at .../res/data/chroma_db (Collection: ipeecs_knowledge_base, Docs: 132)
[INFO] Vector store loaded with 132 document chunks.
[INFO] Bot setup hook initialized.
[INFO] Synced 0 application commands.
[INFO] Bot connected successfully as: IPEECS Advisor Bot#1234 (ID: 1234567890)
[INFO] Bot is ready to accept 1-on-1 DM inquiries.
```

> **提示**：按下 `Ctrl + C` 可觸發 Graceful Shutdown，安全關閉連線與保存狀態。機器人上線時會自動將狀態設為「正在收聽 私訊諮詢系所規章與選課」。

---

## 6. Discord 互動與提問指引

### 如何開始使用
1. 在 Discord 伺服器成員名單中找到機器人。
2. 點擊機器人頭像，選擇 **「發送訊息」** 進入 1 對 1 私訊（DM）視窗。
3. 直接發送任何想詢問的修課、專長學分或系所規章問題。
4. 若在公開頻道中 `@機器人`，機器人會主動提示並引導您至私訊視窗提問。

---

### 提問情境範例

#### 情境 1：明確條件查詢
> **使用者**：請問 113 學年度入學的資工專長，大二必修有哪些？  
> **機器人**：  
> 113 學年度資訊工程專長大二必修課程如下：
> - 資料結構（3學分）
> - 演算法（3學分）
> - 物件導向程式設計（3學分）  
    > ...  
    > 📌 參考來源：`113學年度_資訊工程專長.pdf`

#### 情境 2：籠統問題（機器人主動追問）
> **使用者**：畢業門檻是多少學分？  
> **機器人**：  
> 資電學士班最低畢業學分為 **128 學分**，包含共同必修（25學分）、院訂必修（25學分）、院訂必選（至少6學分）、專長領域必選修及創意創業學分學程等。
>
> 請問您是**哪一學年度入學**？或是目前選擇哪一個**專長領域**（資訊工程／電機工程／通訊工程／網路工程）呢？告訴我後我能為您提供更精確的學分明細喔！

#### 情境 3：多輪連續追問（代名詞自動還原）
> **使用者**：那如果是電機專長呢？  
> **機器人**：（自動結合前述上下文，辨識出是在詢問 113 學年度電機專長畢業學分與專長修課條件並給出精確解答）

#### 情境 4：超出範圍／無資料（安全附帶系辦資訊）
> **使用者**：請問資電學士班學生的宿舍保證住幾年？  
> **機器人**：  
> 我目前在規章資料庫中查無足夠的相關資訊（問題超出規章範圍或查無記錄）。
>
> 若我有無法回答的問題，或是需要進一步協助，也歡迎透過以下方式聯繫系辦公室：  
> 🏢 **資訊電機學院學士班辦公室**  
> 📞 **電話**：03-4227151 分機 35007  
> 📧 **信箱**：ncu35007@ncu.edu.tw  
> 📍 **位置**：工程五館E6 B棟106室 (E6-B106)  
> ⏰ **服務時間**：週一至週五 08:30 - 17:00

---

### 特殊指令

- **`/reset`** 或 輸入 **`重新開始` / `重設` / `reset`**：
  清除當前使用者在機器人中的短期對話記憶，重新開啟新話題。

---

## 7. 系統維護與客製化

### 增加手動 Markdown 文件
若有尚未製作成網頁或 PDF 的常見問答（FAQ），可直接新增 Markdown 檔案至 `res/data/markdown`（例如 `自訂常見問題集.md`），接著執行：
```bash
python sync_data.py --skip-crawl
```
即可立即將自訂 QA 整合至機器人的知識庫中。

---