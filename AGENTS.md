# Code style
- Use NumPy style for docstrings. If any exceptions are raised make sure that they're documented in the "Raises" section.
- When you make changes to the logic of a function or class make sure its docstring is still valid. If required, update the docstrong to match the changes.
- Run `ruff check --fix` to make sure your code is formatter correctly.
- Use single backticks for inline code references in docstrings
- Don't use conditional `if TYPE_CHECKING`
- Always add a `__all__` in `__init__.py` with strings of units that are exported
- Make sure that only objects that are needed outside a module are exposed in the `__init__.py`. Tests don't count as external usage; they should import directly from the submodule if something isn't in `__init__.py`.

# Key Directories
- `alp_data` - main python library

# Environment Setup
- Make sure you use `uv run` for running python commands
