# Tests that Python environment is setup correctly
# Author(s): Cedric Donie (cedricdonie@gmail.com)

import sys


def is_venv():
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def main():
    assert is_venv()
    assert int(sys.version_info.major) == 3
    assert int(sys.version_info.minor) == 6


if __name__ == "__main__":
    main()
