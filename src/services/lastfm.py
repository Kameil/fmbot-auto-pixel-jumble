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
        API_ROOT_URL=" http://ws.audioscrobbler.com/2.0/",
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

            async with self.session.get(
                url=self.API_ROOT_URL, params=params, headers=self.headers
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

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
