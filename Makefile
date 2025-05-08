.PHONY: serve-docs

serve-docs:
	uv run mike serve -b docs-site
