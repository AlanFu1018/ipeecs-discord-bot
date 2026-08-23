"""Chat service integrating query rewriting, RAG retrieval, and LLM answer generation."""
from typing import Any, Dict, List, Optional

from ..core.config import Settings
from ..core.logger import logger
from ..llm_api.llm_base import BaseLLMProvider
from ..rag.vector_store import VectorStore
from .session import SessionManager, UserSession


class ChatService:
    """Core RAG conversational service."""

    def __init__(
        self,
        settings: Settings,
        llm_provider: BaseLLMProvider,
        vector_store: VectorStore,
    ):
        self.settings = settings
        self.llm = llm_provider
        self.vector_store = vector_store
        self.session_manager = SessionManager(
            timeout_minutes=settings.session_timeout_minutes,
            max_history_turns=settings.max_history_turns,
        )
        self.dept_info = settings.department_info

    def get_fallback_message(self, is_error: bool = False) -> str:
        """Standard fallback message when information is not found or an error occurs."""
        name = self.dept_info.get("name", "資訊電機學院學士班辦公室")
        phone = self.dept_info.get("phone", "03-4227151 分機 35007")
        email = self.dept_info.get("email", "ncu35007@ncu.edu.tw")
        location = self.dept_info.get("location", "工程五館E6 B棟106室 (E6-B106)")
        office_hours = self.dept_info.get("office_hours", "週一至週五 08:30 - 17:00")

        contact_text = (
            "若我有無法回答的問題，或是需要進一步協助，也歡迎透過以下方式聯繫系辦公室：\n\n"
            f"🏢 **{name}**\n"
            f"📞 **電話**：{phone}\n"
            f"📧 **信箱**：{email}\n"
            f"📍 **位置**：{location}\n"
            f"⏰ **服務時間**：{office_hours}"
        )

        if is_error:
            return f"抱歉，目前系統處理時發生異常。\n\n{contact_text}"

        return f"我目前在規章資料庫中查無足夠的相關資訊（問題超出規章範圍或查無記載）。\n\n{contact_text}"

    async def rewrite_query(self, session: UserSession, current_query: str) -> str:
        """Condenses multi-turn conversation into a standalone search query."""
        if not session.messages:
            return current_query

        history_text = session.get_history_summary()
        prompt = (
            "【對話歷史】\n"
            f"{history_text}\n\n"
            "【使用者最新輸入】\n"
            f"{current_query}\n\n"
            "【任務】\n"
            "請結合對話歷史，將使用者最新輸入的問句改寫為一個「獨立且語意完整」的單一繁體中文搜尋問句（補齊代名詞、主詞或修課年級等上下文）。"
            "請直接輸出改寫後的問句，不要添加任何引號、解釋或多餘說明。"
        )

        try:
            standalone_query = await self.llm.generate_response(
                prompt=prompt,
                temperature=0.0,
                max_output_tokens=150,
            )
            condensed = standalone_query.strip()
            if condensed:
                logger.info(f"Query rewritten: '{current_query}' -> '{condensed}'")
                return condensed
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original query: {e}")

        return current_query

    def build_system_prompt(self, context_docs: List[Dict[str, Any]]) -> str:
        """Builds system prompt instructions for the LLM."""
        sources_context = ""
        for i, doc in enumerate(context_docs, 1):
            source_name = doc.get("metadata", {}).get("source", "未知來源")
            title = doc.get("metadata", {}).get("title", "")
            content = doc.get("content", "")
            sources_context += f"--- [參考規章 {i}] (來源: {source_name} | {title}) ---\n{content}\n\n"

        dept_name = self.dept_info.get("name", "資訊電機學院學士班辦公室")
        phone = self.dept_info.get("phone", "03-4227151 分機 35007")
        email = self.dept_info.get("email", "ncu35007@ncu.edu.tw")
        location = self.dept_info.get("location", "工程五館E6 B棟106室 (E6-B106)")
        office_hours = self.dept_info.get("office_hours", "週一至週五 08:30 - 17:00")

        return (
            f"你是國立中央大學資訊電機學院學士班（資電學士班）的專屬智能客服與修課規章顧問。\n"
            "請以繁體中文（台灣）為使用者提供親切、專業、精確且有依據的解答。\n\n"
            "【回答守則】\n"
            "1. 【嚴格依據資料】：必須嚴格依據下方提供的【參考規章資料】進行回答，禁止編造或憑空臆測任何未記載的規定、學分數或門檻。\n"
            "2. 【主動提供資料】:若資訊在可以回答的範圍內，且資料庫有答案，請直接幫使用者解惑。 "
            "3. 【主動追問與釐清細節】：若使用者的問題較為籠統或缺乏關鍵條件（例如：尚未指明**入學/適用學年度**、**專長領域**（電機工程/資訊工程/通訊工程/網路工程）、**年級**或**身分**（如轉學生/雙主修）等），導致不同情況適用不同規章時：\n"
            "   - 請先就目前已知資訊提供概括或主流說明。\n"
            "   - **主動親切地追問使用者具體的詳細條件**（例如：「請問您是哪一學年度入學？或是選擇哪個專長領域（資工/電機/通訊/網路）呢？告訴我後我能為您提供更精確的規定喔！」），以利後續給予最精確的解答。\n"
            "4. 【正常回答格式（禁止主動附帶系辦資訊）】：若參考資料中有足夠資訊可回答問題，請精準回答，並在回覆最末尾附註參考來源（例如：`📌 參考來源：[來源檔案名稱或網址]`）。\n"
            "**請注意：在能夠正常回答或追問釐清的情況下，絕對不要附帶系辦公室聯絡資訊（電話/信箱/位置等），保持回覆精簡專業**。\n"
            "5. 【查無資料/超出範圍處理（僅此情況附帶系辦資訊）】：若參考資料中**完全沒有足夠資訊**回答，或是問題超出資電學士班規章範圍，請明確表示查無相關記載，**僅在此種無法回答的情況下**附上以下系辦公室聯絡方式：\n"
            "   若我有無法回答的問題，或是需要進一步協助，也歡迎透過以下方式聯繫系辦公室：\n"
            f"   🏢 {dept_name}\n"
            f"   📞 電話：{phone}\n"
            f"   📧 信箱：{email}\n"
            f"   📍 位置：{location}\n"
            f"   ⏰ 服務時間：{office_hours}\n\n"
            "【參考規章資料】\n"
            f"{sources_context}"
        )

    async def answer_message(self, user_id: str, user_message: str) -> str:
        """Main entry point to process a user message and return an answer."""
        clean_input = user_message.strip()
        session = self.session_manager.get_or_create_session(user_id)

        # Handle reset command
        if clean_input in ["/reset", "重設", "重新開始", "reset"]:
            session.clear()
            return "✅ 已重置您的對話記憶！請問有什麼我可以協助您的系所規章或選課問題嗎？"

        # Step 1: Query Condensing (Rewrite)
        standalone_query = await self.rewrite_query(session, clean_input)

        # Step 2: Vector Retrieval
        docs = await self.vector_store.search(
            query=standalone_query,
            top_k=self.settings.top_k,
        )

        logger.info(f"Retrieved {len(docs)} documents for query: '{standalone_query}'")

        # Step 3: LLM Generation
        if not docs:
            response_text = self.get_fallback_message(is_error=False)
        else:
            system_prompt = self.build_system_prompt(docs)
            user_prompt = f"使用者問題：{clean_input}\n（改寫檢索語意：{standalone_query}）"

            try:
                response_text = await self.llm.generate_response(
                    prompt=user_prompt,
                    system_instruction=system_prompt,
                    temperature=self.settings.llm_temperature,
                    max_output_tokens=self.settings.llm_max_output_tokens,
                )
                if not response_text:
                    response_text = self.get_fallback_message(is_error=False)
            except Exception as e:
                logger.error(f"Error generating LLM response: {e}", exc_info=True)
                response_text = self.get_fallback_message(is_error=True)

        # Step 4: Update Session History
        session.add_message(role="user", content=clean_input, max_turns=self.settings.max_history_turns)
        session.add_message(role="model", content=response_text, max_turns=self.settings.max_history_turns)

        return response_text
