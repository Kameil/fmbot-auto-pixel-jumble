import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

from args import load_args
from inference import InferenceEngine

load_dotenv()


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
        print(f"bot logged in as {self.user.name}")  # type: ignore
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

            def process_img(img_b: bytes):
                query_emb = self.inference_eng.encode(img_b)
                result, score = self.inference_eng.search(query_emb)
                return result, score

            attachment = message.attachments[0]
            img_b = await self.fetch_img(attachment.url)  # type: ignore
            result, _ = await self.loop.run_in_executor(
                self.executor, process_img, img_b
            )
            if result:
                await message.channel.send(result)

    async def on_message(self, message: discord.Message, /) -> None:
        if message.author.bot:
            print(message.author.id, message.guild.id)
            if (
                message.author.id == 356268235697553409
                and message.guild.id == self.target_guild_id
            ):
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
        else:
            await self.process_commands(message)

    async def on_message_edit(
        self, before, message: discord.Message
    ):  # message como after
        if message.author.bot:
            print(message.author.id, message.guild.id)
            if (
                message.author.id == 356268235697553409
                and message.guild.id == self.target_guild_id
            ):
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
