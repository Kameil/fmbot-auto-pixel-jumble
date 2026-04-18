from typing import Dict
import aiohttp

import asyncio
from dataclasses import dataclass


@dataclass
class Album:
    name: str
    small: str
    mbid: str
    mb_img: str
    medium: str
    large: str
    extralarge: str


class UserFm:
    def __init__(
        self,
        user_name: str,
        session: aiohttp.ClientSession,
        api_key: str,
        user_agent: str,
        API_ROOT_URL="http://ws.audioscrobbler.com/2.0/",
    ):
        self.session = session
        self.API_ROOT_URL = API_ROOT_URL
        self.user_name = user_name
        self.headers = {"User-Agent": user_agent}
        self.api_key = api_key

    # JSON: /2.0/?method=user.gettopalbums&user=rj&api_key=YOUR_API_KEY&format=json
    # https://www.last.fm/api/show/user.getTopAlbums

    async def get_all_albums(self) -> list[Album]:
        page = 1
        all_albums: list[Album] = []
        max_retries = 5
        while True:
            params = {
                "method": "user.gettopalbums",
                "user": self.user_name,
                "api_key": self.api_key,
                "format": "json",
                "period": "overall",
                "limit": 1000,
                "page": page,
            }
            sucess = False
            data = None
            for attemp in range(max_retries):
                wait_time = 2**attemp  # Exponential backoff
                try:
                    async with self.session.get(
                        url=self.API_ROOT_URL, params=params, headers=self.headers
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            sucess = True
                            break
                        else:
                            if resp.status == 404:
                                if page == 1:
                                    raise RuntimeError(
                                        f'Page {page} not found (404). Stopping retries. Maybe user "{self.user_name}" does not exist?'
                                    )
                            print(
                                f"failed to fetch page {page}, status: {resp.status}. Retrying in {wait_time}s..."
                            )
                            if resp.status in [
                                429,
                                503,
                            ]:  # Too Many Requests or Service Unavailable
                                await asyncio.sleep(wait_time)  # Exponential backoff
                            else:
                                await asyncio.sleep(0.2)  # Short delay for other errors
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"Error on page {page}: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)  # Exponential backoff
            if not sucess or data is None:
                raise RuntimeError(
                    f"Failed to fetch page {page} after {max_retries} attempts."
                )

            albums = data["topalbums"]["album"]

            for a in albums:
                images = {img["size"]: img["#text"] for img in a["image"]}
                mb_id = a.get("mbid", "")
                album = Album(
                    name=a["name"],
                    mbid=mb_id,
                    mb_img=f"https://coverartarchive.org/release/{mb_id}/front"
                    if mb_id
                    else "",
                    small=images.get("small", ""),
                    medium=images.get("medium", ""),
                    large=images.get("large", ""),
                    extralarge=images.get("extralarge", ""),
                )

                all_albums.append(album)

            attr = data["topalbums"]["@attr"]
            total_pages = int(attr["totalPages"])

            print(f"page {page}/{total_pages} coleted")

            if page >= total_pages:
                break

            page += 1
            await asyncio.sleep(0.2)

        return all_albums
