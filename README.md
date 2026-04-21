# esp-data
This repository contains data tools for managing data operations in ESP's AI / science projects.

## Help

See [documentation](http://esp/docs/esp-data).

> [!NOTE]
> You need to be connected to Tailscale for the link to work.

## Development Setup

1. Install dependencies including dev tools:
   ```
   uv sync --dev
   ```
2. Set up pre-commit hooks:
   ```
   # Append '--overwrite' to overwrite existing hooks if you have them for e.g. from 'pre-commit' lib
   uv run prek install
   ```

## Running tests

```
uv run pytest
```

## Serving Documentation Locally

To preview the documentation site locally, use the following command:

```sh
make serve-local-docs
```

This will start a local server using [mike](https://github.com/jimporter/mike) and serve the docs at the default address (usually http://localhost:8000). The docs will be built from the `docs-site` branch.
