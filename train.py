import asyncio
import hashlib
import os
from io import BytesIO

import aiohttp
from numpy import trace
import torch
import torch.nn as nn
import torch.nn.functional as F  # o coisa aí
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
from dotenv import load_dotenv
import traceback

from lastfm import Album, UserFm
from args import load_args

load_dotenv()
args = load_args()


def pixelate_8x8(img: Image.Image) -> Image.Image:
    small = img.resize((8, 8), resample=Image.BILINEAR)  # type: ignore
    big = small.resize((256, 256), resample=Image.NEAREST)  # type: ignore

    return big


def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest() + ".jpg"


class Trainer:
    def __init__(self, username: str, cache_dir="cache") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Lambda(pixelate_8x8),
                transforms.Resize((8, 8)),
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
            timeout = aiohttp.ClientTimeout(total=60)
            try:
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.read()
            except (asyncio.TimeoutError, aiohttp.ClientError):
                return None
            try:
                img = Image.open(BytesIO(data)).convert("RGB")
                is_music_brainz = "coverartarchive.org" in url
                format = "JPEG"
                if is_music_brainz:
                    format = "PNG"
                img = img.resize(
                    (8, 8),
                    resample=Image.BILINEAR,  # type: ignore
                )  # salvar no tamanho da mobilenet se não o cache fica gigante
                buffer = BytesIO()
                img.save(buffer, format=format)
                data = buffer.getvalue()
            except Exception:
                traceback.print_exc()
                return None

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
        tensor = self.transform(img).to(self.device)
        # batch = self.transform(img).unsqueeze(0).to(self.device)  # type: ignore
        # with torch.no_grad():
        #     emb = self.model(batch)
        # emb = emb.squeeze()
        mean_color = tensor.mean(dim=(1, 2))  # [3]
        flat = tensor.flatten()  # [3*H*W]
        emb = torch.cat(
            [flat * 0.5, mean_color * 2.0]
        )  # levar a cor mais em consideração
        emb = F.normalize(emb, dim=0)
        return emb

    def get_embeddings_batch(self, imgs_bytes: list[bytes]) -> torch.Tensor:
        tensors = [
            self.transform(Image.open(BytesIO(b)).convert("RGB")) for b in imgs_bytes
        ]

        batch = torch.stack(tensors).to(self.device)  # [B, 3, H, W]

        mean_color = batch.mean(dim=(2, 3))  # [B, 3]
        flat = batch.flatten(start_dim=1)  # [B, N]

        emb = torch.cat([flat * 0.5, mean_color * 2.0], dim=1)
        emb = F.normalize(emb, dim=1)

        return emb

    async def build_album_embeddings(self, session, albums: list[Album]):
        results: dict[str, torch.Tensor] = {}

        batch_size = 32
        buffer_imgs: list[bytes] = []
        buffer_keys: list[str] = []

        async def fetch(album: Album):
            results = []

            urls = []

            if album.extralarge and album.extralarge.startswith("http"):
                urls.append(album.extralarge)

            if getattr(album, "mb_img", None):
                urls.append(album.mb_img)

            for url in urls:
                img_bytes = await self.get_image_bytes(session, url)

                if not img_bytes:
                    continue

                key = f"{album.name}::{url}"
                results.append((key, img_bytes))

            return results if results else None

        tasks = [fetch(album) for album in albums]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            items = await coro

            if not items:
                continue

            for key, img_bytes in items:
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
        username = args.username
        print("training for " + username)
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
    except Exception:
        traceback.print_exc()
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
