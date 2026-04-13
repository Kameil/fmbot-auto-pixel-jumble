import argparse


def load_args():
    """
    --username  dest username
    --cache-dir dest cache_dir
    """
    argument_parser = argparse.ArgumentParser(description="fmbot auto jumble")
    argument_parser.add_argument(
        "-u",
        "--username",
        type=str,
        help="lastfm username",
        dest="username",
        required=True,
    )
    argument_parser.add_argument(
        "--cache-dir",
        type=str,
        help="album image cache dir",
        default="cache",
        dest="cache_dir",
    )
    argument_parser.add_argument(
        "-g",
        "--target-guild",
        type=int,
        help="guild_id",
        dest="target_guild_id",
        default=941365514129776710,
    )
    return argument_parser.parse_args()
