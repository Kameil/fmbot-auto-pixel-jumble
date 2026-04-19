import asyncio
import os
import traceback

import aiohttp
from dotenv import load_dotenv

from src.core.trainer import Trainer
from src.services.lastfm import UserFm
from src.utils.args import load_args

args = load_args()
load_dotenv()


async def main():
    session = aiohttp.ClientSession()
    try:
        if os.getenv("LASTFM_API_KEY", "") == "":
            raise ValueError("LASTFM_API_KEY is not set in environment variables.")
        username = args.username.lower()
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
        embds.save(os.path.join("users", f"{username}.pt"))
    except Exception:
        traceback.print_exc()
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
