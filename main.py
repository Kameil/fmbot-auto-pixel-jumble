from dotenv import load_dotenv
from src.utils.args import load_args
from src.bot.client import MyBot
import os

load_dotenv()

if __name__ == "__main__":
    if os.getenv("DISCORD_TOKEN", False) is None:
        raise ValueError("DISCORD_TOKEN environment variable is not set")

    args = load_args()
    path = f"{args.username}.pt"
    assert os.path.exists(path), f"{path}.pt does not exists"
    bot = MyBot(path, target_guild_id=args.target_guild_id)
    bot.run(os.getenv("DISCORD_TOKEN", ""))
