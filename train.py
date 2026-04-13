import asyncio
import hashlib
import os
from io import BytesIO

import aiohttp
import torch
import torch.nn as nn
import torch.nn.functional as F  # o coisa aí
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
from dotenv import load_dotenv
import traceback

from lastfm import Album, UserFm

load_dotenv()


def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest() + ".jpg"


class Trainer:
    def __init__(self, username: str, cache_dir="cache") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.username = username
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.model = self.load_model()
        self.semaphore = asyncio.Semaphore(10)

    class ComputedEmbeddings:
        def __init__(self, embs: dict[str, torch.Tensor]) -> None:
            self.embeddings = embs

        def save(self, path: str):
            torch.save({"embeddings": self.embeddings}, path)

    async def get_image_bytes(self, session: aiohttp.ClientSession, url):
        async with self.semaphore:
            filename = url_to_filename(url)
            path = os.path.join(self.cache_dir, filename)

            # ja existe → usa cache
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()

            # nao existe → baixa
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()

            # salva no cache
            with open(path, "wb") as f:
                f.write(data)

            return data

    def load_model(self) -> models.MobileNetV2:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = nn.Identity()  # type: ignore vai se fuder pyright n quero saber
        model.eval()
        model.to(self.device)
        return model

    def get_embedding(self, img_bytes: bytes) -> torch.Tensor:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        batch = self.transform(img).unsqueeze(0).to(self.device)  # type: ignore
        with torch.no_grad():
            emb = self.model(batch)
        emb = emb.squeeze()
        emb = F.normalize(emb, dim=0)
        return emb

    def get_embeddings_batch(self, imgs_bytes: list[bytes]) -> torch.Tensor:
        imgs = [
            self.transform(Image.open(BytesIO(b)).convert("RGB")) for b in imgs_bytes
        ]

        batch = torch.stack(imgs).to(self.device)  # type: ignore unknown é minha pomba

        with torch.no_grad():
            embs = self.model(batch)

        embs = F.normalize(embs, dim=1)
        return embs

    async def build_album_embeddings(self, session, albums: list[Album]):
        results: dict[str, torch.Tensor] = {}

        batch_size = 32
        buffer_imgs: list[bytes] = []
        buffer_keys: list[str] = []

        async def fetch(album: Album):
            if (
                not album.extralarge
                or not album.extralarge.startswith("http")
                or album.extralarge == ""
            ):
                return None

            img_bytes = await self.get_image_bytes(session, album.extralarge)

            if img_bytes is None:
                return None

            key = f"{album.name}::{album.extralarge}"
            return key, img_bytes

        tasks = [fetch(album) for album in albums]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            item = await coro

            if item is None:
                continue

            key, img_bytes = item
            if not img_bytes:
                continue

            buffer_imgs.append(img_bytes)
            buffer_keys.append(key)

            if len(buffer_imgs) >= batch_size:
                embs = self.get_embeddings_batch(buffer_imgs)

                for k, e in zip(buffer_keys, embs):
                    results[k] = e.cpu()

                buffer_imgs.clear()
                buffer_keys.clear()

        # resto final
        if buffer_imgs:
            embs = self.get_embeddings_batch(buffer_imgs)

            for k, e in zip(buffer_keys, embs):
                results[k] = e.cpu()

        return self.ComputedEmbeddings(results)


async def main():
    session = aiohttp.ClientSession()
    try:
        username = "racomatavo"
        user = UserFm(
            username,
            session=session,
            api_key=os.getenv("LASTFM_API_KEY", ""),
            user_agent=os.getenv("USERAGENT", ""),
        )
        albums = await user.get_all_albums()
        trainer = Trainer(username=f"{username}")
        embds = await trainer.build_album_embeddings(session, albums)
        embds.save(f"{username}.pt")
    except:
        traceback.print_exc()
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
