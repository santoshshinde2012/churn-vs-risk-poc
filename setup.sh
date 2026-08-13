#!/usr/bin/env bash
# One-time setup: create .venv and install the pinned dependencies.
#
# The pins in requirements.txt are deliberate - the README quotes exact numbers,
# and "everything is seeded" is only true if the libraries are too. They have
# wheels for CPython 3.10 through 3.13. On a newer interpreter pip falls back to
# building from source, which fails on matplotlib, so this script looks for a
# supported one rather than letting that happen fifty lines into the output.
#
# Override with:  PYTHON=/path/to/python3.12 ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

MIN_MINOR=10
MAX_MINOR=13

supported() {
    "$1" -c "import sys; sys.exit(0 if (3, $MIN_MINOR) <= sys.version_info < (3, $((MAX_MINOR + 1))) else 1)" \
        >/dev/null 2>&1
}

if [ -n "${PYTHON:-}" ]; then
    if ! supported "$PYTHON"; then
        echo "PYTHON=$PYTHON is $("$PYTHON" --version 2>&1), outside 3.$MIN_MINOR-3.$MAX_MINOR." >&2
        echo "Either point PYTHON at a supported interpreter, or loosen requirements.txt" >&2
        echo "and accept that the numbers in the README may shift." >&2
        exit 1
    fi
else
    PYTHON=""
    for candidate in python3.12 python3.11 python3.13 python3.10 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && supported "$candidate"; then
            PYTHON="$candidate"
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "No CPython between 3.$MIN_MINOR and 3.$MAX_MINOR found on PATH." >&2
        echo "Default python3 is: $(python3 --version 2>&1 || echo 'not found')" >&2
        echo "Install one (brew install python@3.12) or set PYTHON=/path/to/python3.12" >&2
        exit 1
    fi
fi

echo "using $PYTHON ($("$PYTHON" --version 2>&1))"
rm -rf .venv
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

echo
echo "done. now run:"
echo "    .venv/bin/python run_all.py"
