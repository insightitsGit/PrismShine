"""Allow ``python -m prismshine …`` as well as the ``prismshine`` console script."""

from prismshine.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
