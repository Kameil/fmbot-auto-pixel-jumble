# [fmbot](https://github.com/fmbot-discord/fmbot) auto pixel jumble

- A selfbot that automatically solves pixel jumble game from FMbot.

## how to use

- clone repo
- sync deps

```
uv sync --frozen
```

- create .env

```
mv .env.example .env 
```

```
nvim .env 
```

- train model

```
uv run train.py -u <username>
```

- After the model is trained, you can clear the training cache with `rm -r ./cache`.

- run bot

```
uv run main.py -u <username> -g <guild_id>
```

- This bot was created for study purposes; use at your own risk.
