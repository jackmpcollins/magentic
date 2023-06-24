# agentic

## Type Checking

Many type checkers will raise warnings or errors for functions with the `prompt` decorator due to the function having no body or return value. There are several ways to deal with these.

1. Disable the check globally for the type checker. For example in mypy by disabling error code `empty-body`.
1. Make the function body `...` (does not work for mypy) or `raise`.
   ```python
   @prompt
   def random_color() -> str:
       """Choose a color"""
       ...
   ```
1. Use comment `# type: ignore[empty-body]` on each function.
   ```python
   @prompt
   def random_color() -> str:  # type: ignore[empty-body]
       """Choose a color"""
   ```
