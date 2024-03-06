# Type Checking

Many type checkers will raise warnings or errors for functions with the `@prompt` decorator due to the function having no body or return value. There are several ways to deal with these.

1. Disable the check globally for the type checker. For example in mypy by disabling error code `empty-body`.
   ```toml
   # pyproject.toml
   [tool.mypy]
   disable_error_code = ["empty-body"]
   ```
1. Make the function body `...` (this does not satisfy mypy) or `raise`.
   ```python
   @prompt("Choose a color")
   def random_color() -> str: ...
   ```
1. Use comment `# type: ignore[empty-body]` on each function. In this case you can add a docstring instead of `...`.
   ```python
   @prompt("Choose a color")
   def random_color() -> str:  # type: ignore[empty-body]
       """Returns a random color."""
   ```
