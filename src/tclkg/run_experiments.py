"""Compatibility entry point for the parameter sweep launcher."""

from .launcher import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
