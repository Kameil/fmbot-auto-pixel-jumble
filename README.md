# [fmbot](https://github.com/fmbot-discord/fmbot) auto pixel jumble

- A bot that automatically solves pixel jumble game from FMbot.

## how to use

- clone repo
- sync deps

```
uv sync --frozen
```

- train model

```
uv run train.py --u <username>
```

- run bot

```
uv run main.py -u <username> -g <guild_id>
```

- This bot was created for study purposes; use at your own risk.
