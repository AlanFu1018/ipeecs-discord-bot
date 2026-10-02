import logging

# 1. 基本設定：設定最低顯示等級為 INFO，並定義輸出格式 (加入時間與等級)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 2. 開始寫日記
logging.debug("這行是 DEBUG，因為最低設定是 INFO，所以這行不會印出來！")
logging.info("程式開始執行囉")
logging.warning("警告：找不到外部檔案，將使用預設設定")
logging.error("糟糕，發生了錯誤，但程式會繼續跑")