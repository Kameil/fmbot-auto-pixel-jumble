import logging
import os

from dotenv import load_dotenv

from src.bot.client import MyBot
from src.utils.args import load_args

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

if __name__ == "__main__":
    if os.getenv("DISCORD_TOKEN", False) is None:
        raise ValueError("DISCORD_TOKEN environment variable is not set")

    args = load_args()
    path = f"{args.username}.pt"
    assert os.path.exists(path), f"{path}.pt does not exists"
    bot = MyBot(path, target_guild_id=args.target_guild_id)
    bot.run(os.getenv("DISCORD_TOKEN", ""))
