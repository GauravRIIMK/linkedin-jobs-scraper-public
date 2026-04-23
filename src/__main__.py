"""Entrypoint so `python -m src ...` works."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
