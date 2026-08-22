"""Discord Bot implementation specialized for 1-on-1 DM consultation."""
import asyncio
from typing import List
import discord
from discord.ext import commands

from ..core.config import Settings
from ..core.logger import logger
from ..services.chat_service import ChatService


def split_message(text: str, limit: int = 1900) -> List[str]:
    """Splits a long response into Discord-compatible chunks (<2000 characters)."""
    if len(text) <= limit:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > limit:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # If a single line itself exceeds limit, split by chars
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            current_chunk = line
        else:
            current_chunk += ("\n" if current_chunk else "") + line

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


class IpeecsDiscordBot(commands.Bot):
    """Department Advisor Discord Bot."""

    def __init__(self, settings: Settings, chat_service: ChatService):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        intents.guild_messages = True

        super().__init__(
            command_prefix=settings.command_prefix,
            intents=intents,
            help_command=None,
        )
        self.settings = settings
        self.chat_service = chat_service

    async def setup_hook(self) -> None:
        """Register slash commands or hooks."""
        logger.info("Bot setup hook initialized.")
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} application commands.")
        except Exception as e:
            logger.warning(f"Failed to sync slash commands: {e}")

    async def on_ready(self) -> None:
        """Called when the bot is connected and ready."""
        logger.info(f"Bot connected successfully as: {self.user} (ID: {self.user.id})")
        logger.info("Bot is ready to accept 1-on-1 DM inquiries.")
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name="私訊諮詢系所規章與選課",
        )
        await self.change_presence(activity=activity)

    async def on_message(self, message: discord.Message) -> None:
        """Handles incoming messages and routes DM conversations."""
        # Ignore messages sent by the bot itself
        if message.author == self.user:
            return

        # Check if the message is in a DM channel
        is_dm = isinstance(message.channel, discord.DMChannel)

        if not is_dm:
            # If user mentioned bot in a guild channel, provide polite DM instruction
            if self.user in message.mentions:
                await message.reply(
                    f"您好！為了維護隱私與頻道整潔，請直接**發送私人訊息（DM）**給我進行一對一系所諮詢喔！😊"
                )
            return

        user_id = str(message.author.id)
        user_text = message.content.strip()

        if not user_text:
            return

        logger.info(f"Received DM from {message.author} ({user_id}): {user_text}")

        # Send typing indicator while querying RAG and generating answer
        try:
            async with message.channel.typing():
                response = await self.chat_service.answer_message(
                    user_id=user_id,
                    user_message=user_text,
                )

            # Split message if it exceeds Discord's 2000 character limit
            parts = split_message(response)
            for part in parts:
                await message.channel.send(part)

        except Exception as e:
            logger.error(f"Error answering message for user {user_id}: {e}", exc_info=True)
            await message.channel.send("抱歉，處理您的訊息時發生錯誤，請稍後再試。")
