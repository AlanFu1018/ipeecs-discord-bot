# 國立中央大學資電學士班智能客服機器人 — 使用手冊 (User Guide)

歡迎使用 **國立中央大學資訊電機學院學士班（IPEECS）智能客服機器人**。本手冊旨在提供使用者、助教及系統管理員完整的操作與維運指引，涵蓋環境部署、資料庫同步、Discord 互動操作及常見問題排解。

---

## 📖 目錄
1. [系統概述與核心功能](#1-系統概述與核心功能)
2. [環境需求與前置準備](#2-環境需求與前置準備)
3. [安裝與設定指南](#3-安裝與設定指南)
4. [知識庫資料同步 (sync_data.py)](#4-知識庫資料同步-sync_datapy)
5. [啟動與運行機器人 (main.py)](#5-啟動與運行機器人-mainpy)
6. [Discord 互動與提問指引](#6-discord-互動與提問指引)
7. [常見問題排解 (FAQ)](#7-常見問題排解-faq)
8. [系統維護與客製化](#8-系統維護與客製化)

---

## 1. 系統概述與核心功能

本機器人專為中央大學資電學士班設計，基於 **檢索增強生成（RAG）** 技術與 **Google Gemini LLM**，提供以下核心價值：

- **一對一私訊諮詢 (DM)**：所有對話皆在私訊中進行，保護學生隱私且不干擾公開伺服器。
- **高精確度規章檢索**：自動爬取並索引 109～114 學年度各專長課規 PDF 與系網資訊，所有回答皆有依據。
- **多輪對話改寫 (Query Condensing)**：能理解上下文代名詞（例如接續問「那大三呢？」、「如果抵免的話呢？」），精準改寫問句後檢索。
- **主動追問與引導**：當使用者問題較為籠統（如未指明學年度或專長領域）時，機器人會主動追問細節以提供最精確答案。
- **嚴謹的 Fallback 機制**：遇查無資料、超出範圍或系統異常時，嚴禁模型幻覺，並統一附上系辦公室聯絡方式。
- **模組化 Adapter 設計**：支援隨時抽換 LLM（Gemini/OpenAI/本地模型）與 Embedding 模組。

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
   - 前往 [Google AI Studio](https://aistudio.google.com/) 申請免費 API Key。

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
  session_timeout_minutes: 15     # 對話閒置過期時間 (分鐘)
  max_history_turns: 5           # 記憶對話輪數上限

llm:
  provider: "gemini"             # gemini / openai / local
  model: "gemini-2.0-flash"      # LLM 模型名稱
  temperature: 0.2
  max_output_tokens: 1500

embedding:
  provider: "gemini"             # gemini / local
  model: "text-embedding-004"    # Embedding 模型
  dimension: 768

rag:
  top_k: 3                       # 檢索前 K 個相關片段
  chunk_size: 600                # 文件分塊字元數
  chunk_overlap: 100             # 分塊重疊字元數
  collection_name: "ipeecs_knowledge_base"

department_info:
  name: "資訊電機學院學士班辦公室"
  phone: "03-4227151 分機 35007"
  email: "ncu35007@ncu.edu.tw"
  location: "工程五館E6 B棟106室 (E6-B106)"
  office_hours: "週一至週五 08:30 - 17:00"
```

---

## 4. 知識庫資料同步 (sync_data.py)

當系所規章有更新、或初次部署專案時，需執行 `sync_data.py` 建立／更新向量資料庫：

```bash
# 完整同步：爬取系網頁面、下載最新規章 PDF 並寫入向量庫
python sync_data.py
```

### 進階參數說明

| 參數 | 說明 | 適用情境 |
| :--- | :--- | :--- |
| *(無參數)* | 執行完整爬蟲 + 解析分塊 + 清除舊庫並重新建立索引。 | 定期規章大改版或初次建庫。 |
| `--skip-crawl` | **略過網路爬蟲**，僅重新解析本機 `res/data/raw/` 與 `res/data/markdown/` 檔案並建庫。 | 已手動加入新 PDF 或修訂 Markdown 時。 |
| `--no-reset` | **不清空現有資料庫**，直接將新分塊追加（Upsert）進 ChromaDB。 | 單純擴充資料，保留原有向量時。 |

範例：
```bash
# 僅重新解析本機文件並建立向量庫
python sync_data.py --skip-crawl
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
[INFO] Initialized VectorStore at .../res/data/chroma_db (Collection: ipeecs_knowledge_base, Docs: 56)
[INFO] Vector store loaded with 56 document chunks.
[INFO] Logged in as IPEECS Advisor Bot#1234 (ID: 1234567890)
[INFO] Bot is ready and listening for Direct Messages (DM)!
```

> **提示**：按下 `Ctrl + C` 可觸發 Graceful Shutdown，安全關閉連線與保存狀態。

---

## 6. Discord 互動與提問指引

### 如何開始使用
1. 在 Discord 伺服器成員名單中找到機器人。
2. 點擊機器人頭像，選擇 **「發送訊息」** 進入 1 對 1 私訊視窗。
3. 直接發送任何想詢問的修課或系所問題。

---

### 提問情境範例

#### 情境 1：明確條件查詢
> **使用者**：請問 113 學年度入學的資工專長，大二必修有哪些？  
> **機器人**：  
> 113 學年度資訊工程專長大二必修課程如下：  
> 1. 資料結構（3學分）  
> 2. 演算法（3學分）  
> 3. 物件導向程式設計（3學分）  
> ...  
> 📌 參考來源：`113學年度_資訊工程專長.pdf`

#### 情境 2：籠統問題（機器人主動追問）
> **使用者**：畢業門檻是多少學分？  
> **機器人**：  
> 資電學士班最低畢業學分為 **128 學分**，包含共同必修（25學分）、院訂必修（25學分）、院訂必選（至少6學分）、專長領域必選修及創意創業學分學程等。  
>   
> 請問您是**哪一學年度入學**？或是目前選擇哪一個**專長領域**（資訊工程／電機工程／通訊工程／網路工程）呢？告訴我後我能為您提供更精確的學分明細喔！

#### 情境 3：多輪連續追問
> **使用者**：那如果是電機專長呢？  
> **機器人**：（自動結合前述上下文，辨識出是在詢問電機專長畢業學分與專長修課條件並給出精確解答）

#### 情境 4：超出範圍／無資料
> **使用者**：請問資電學士班學生的宿舍保證住幾年？  
> **機器人**：  
> 我目前在規章資料庫中查無足夠的相關資訊（問題超出規章範圍或查無記載）。  
>   
> 若有我無法回答的問題，或是需要進一步協助，也歡迎透過以下方式聯繫系辦公室：  
> 🏢 **資訊電機學院學士班辦公室**  
> 📞 **電話**：03-4227151 分機 35007  
> 📧 **信箱**：ncu35007@ncu.edu.tw  
> 📍 **位置**：工程五館E6 B棟106室 (E6-B106)

---

### 特殊指令

- **`/reset`** 或 輸入 **`重新開始` / `重設`**：
  清除當前使用者在機器人中的短期對話記憶，重新開啟新話題。

---

## 7. 常見問題排解 (FAQ)

### Q1: 機器人在 Discord 上顯示離線或沒有回應？
1. **檢查 Token**：確認 `.env` 中的 `DISCORD_BOT_TOKEN` 是否正確填寫且無多餘引號或空白。
2. **檢查 Privileged Intents**：確認 Discord Developer Portal 中的 **Message Content Intent** 是否已開啟。
3. **確認是否在私訊中提問**：本機器人預設僅處理 **DM（一對一私訊）**，不處理群組伺服器公開頻道的閒聊。

### Q2: 執行 `sync_data.py` 時報錯 `GEMINI_API_KEY is not configured`？
- 請在 `.env` 中設定 `GEMINI_API_KEY="你的Key"`。

### Q3: 機器人回答內容與最新法規不符？
- 可能是學校系網發布了新版本 PDF。
- 請確認 `config/urls.txt` 連結是否最新，並重新執行 `python sync_data.py` 更新本地向量資料庫。

---

## 8. 系統維護與客製化

### 調整目標爬蟲網址 (`config/urls.txt`)
在 `config/urls.txt` 中加入新的系網頁面或教務規章網址，以 `//` 作為註解：
```text
// 資電學士班重要規章
最新法規頁面: https://www.ipeecs.ncu.edu.tw/...
```

### 增加手動 Markdown 文件
若有尚未製作成網頁或 PDF 的常見問答（FAQ），可直接新增 Markdown 檔案至 `res/data/markdown/`（例如 `常見問題集.md`），接著執行：
```bash
python sync_data.py --skip-crawl
```
即可立即將自訂 QA 整合至機器人的大腦中。

---
*手冊維護人員：國立中央大學資訊電機學院學士班 開發團隊*
