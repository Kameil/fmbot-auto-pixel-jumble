import logging
import os

from dotenv import load_dotenv

from src.bot.client import MyBot
from src.utils.args import load_args

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logging.getLogger("discord").propagate = False

args = load_args()

if __name__ == "__main__":
    if os.getenv("DISCORD_TOKEN", False) is None:
        raise ValueError("DISCORD_TOKEN environment variable is not set")
    username = args.username
    path = os.path.join("users", f"{username}.pt")
    if os.path.exists(path):
        logging.info(f"Loading model for user {username} from {path}")
        bot = MyBot(path, target_guild_id=args.target_guild_id)
        bot.run(os.getenv("DISCORD_TOKEN", ""))
