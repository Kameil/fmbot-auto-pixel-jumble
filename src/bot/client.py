import asyncio
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import discord
from discord.ext import commands
from src.core.engine import InferenceEngine


class MyBot(commands.Bot):
    def __init__(self, embeddings_path: str, target_guild_id: int) -> None:
        super().__init__(
            command_prefix=".",
            help_command=None,
        )
        assert os.path.exists(embeddings_path)
        self.target_user_id = 356268235697553409  # .fmbot ID constant
        self.inference_eng = InferenceEngine(embeddings_path)
        self.target_guild_id = target_guild_id
        self.queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.session: aiohttp.ClientSession | None = None
        self.logger = logging.getLogger("Bot")

    async def headers_context(self):
        async with aiohttp.ClientSession() as session:
            return await discord.HeadersContext.desktop(session)

    async def setup_hook(self) -> None:
        self.session = aiohttp.ClientSession()

    async def on_ready(self):
        self.logger.info(f"bot logged in as {self.user.name}")  # type: ignore
        self.loop.create_task(self.process_queue())

    async def fetch_img(self, url: str) -> bytes | None:
        if self.session is not None:
            try:
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.read()
            except Exception as e:
                self.logger.error(f"error fetching fmbot img {str(e)}")
                traceback.print_exc()
        self.logger.warning("http session does not exists")
        return None

    async def process_queue(self):
        while True:
            message = await self.queue.get()
            message: discord.Message

            async with message.channel.typing():

                def process_img(img_b: bytes):
                    query_emb = self.inference_eng.encode(img_b)
                    result, score = self.inference_eng.search(query_emb)
                    return result, score

                attachment = message.attachments[0]
                img_b = await self.fetch_img(attachment.url)
                if img_b is None:
                    return
                result, score = await self.loop.run_in_executor(
                    self.executor, process_img, img_b
                )

                def label_parsing(label: str) -> tuple[str, str]:
                    """
                    receives "album_name::album_image_link"
                    returns (album_name, album_image_url)
                    """
                    vars = label.rsplit("::")
                    return vars[0], vars[1]

                human_score = score * 100
                if result:
                    album_name, album_image_link = label_parsing(result)
                    if score > 0.9:  # maior do que 90% de similaridade
                        self.logger.info(
                            f"{attachment.url} got {human_score:.2f}% {album_name}::{album_image_link}"
                        )
                        await message.channel.send(album_name.lower())
                    else:
                        self.logger.warning(
                            f"enought confidence: {human_score if result else 0:.2f}% \n {attachment.url} got {album_name}::{album_image_link}"
                        )
                else:
                    self.logger.warning(f"{attachment.url} got no result")

    async def process_fmbot_message(self, message: discord.Message):
        if not message.attachments:
            return
        attachments = message.attachments
        embeds = message.embeds
        if attachments and embeds:
            embed = embeds[0]
            if (
                embed.fields
                and embed.fields[0].name
                and "Pixel Jumble" in embed.fields[0].name
            ):
                cor = embed.color
                if cor and cor.value == 2021217:
                    return  # for
                await self.queue.put(message)

    async def on_message(self, message: discord.Message, /) -> None:
        if message.author.bot:
            if (
                message.author.id == 356268235697553409
                and message.guild.id == self.target_guild_id  # type: ignore
            ):
                await self.process_fmbot_message(message)
        else:
            await self.process_commands(message)

    async def on_message_edit(
        self, before: discord.Message, message: discord.Message
    ):  # message como after
        if message.author.bot:
            if (
                message.author.id == 356268235697553409
                and message.guild.id == self.target_guild_id  # type: ignore
            ):
                await self.process_fmbot_message(message)

    async def close(self):
        if self.session is not None:
            await self.session.close()
        await super().close()
