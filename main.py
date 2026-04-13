import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import logging

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

from args import load_args
from inference import InferenceEngine

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MyBot(commands.Bot):
    def __init__(self, embeddings_path: str, target_guild_id: int) -> None:
        super().__init__(
            command_prefix=".",
            help_command=None,
        )
        assert os.path.exists(embeddings_path)
        self.target_user_id = 356268235697553409  # não tem problema ser hardcoded já que nao vai mudar, mesmod dando toc
        self.inference_eng = InferenceEngine(embeddings_path)
        self.target_guild_id = target_guild_id
        self.queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.session: aiohttp.ClientSession | None = None

    async def headers_context(self):
        async with aiohttp.ClientSession() as session:
            return await discord.HeadersContext.desktop(session)

    async def setup_hook(self) -> None:
        self.session = aiohttp.ClientSession()

    async def on_ready(self):
        logging.info(f"bot logged in as {self.user.name}")  # type: ignore
        self.loop.create_task(self.process_queue())

    async def fetch_img(self, url: str) -> bytes:
        assert self.session is not None

        async with self.session.get(url) as resp:
            resp.raise_for_status()
            return await resp.read()

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
                img_b = await self.fetch_img(attachment.url)  # type: ignore
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

                if result and score > 0.9:  # maior do que 90% de similaridade
                    album_name, album_image_link = label_parsing(result)
                    logging.info(
                        f"{attachment.url} got {score:.2f}% {album_name}::{album_image_link}"
                    )
                    await message.channel.send(album_name.lower())

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


if __name__ == "__main__":
    args = load_args()
    path = f"{args.username}.pt"
    assert os.path.exists(path), f"{path}.pt does not exists"
    bot = MyBot(path, target_guild_id=args.target_guild_id)
    bot.run(os.getenv("DISCORD_TOKEN", ""))
