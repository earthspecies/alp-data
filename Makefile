.PHONY: serve-remote-docs serve-local-docs

serve-remote-docs:
	git fetch origin
	uv run mike serve -b origin/docs-site

serve-local-docs:
	uv run mkdocs serve
