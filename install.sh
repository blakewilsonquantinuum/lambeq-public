#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo 'Usage: install.sh VIRTUALENV_DIRECTORY'
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR=$1

python3 -m venv $VENV_DIR
$VENV_DIR/bin/pip install --upgrade pip
$VENV_DIR/bin/pip install cython numpy
$VENV_DIR/bin/pip install "$SCRIPT_DIR"

function download_depccg_parser { $VENV_DIR/bin/python -m depccg en download; }
if download_depccg_parser; then
    true  # success, no further action needed
else
    # See README.md for rationale
    $VENV_DIR/bin/pip install --upgrade typing-extensions
    download_depccg_parser
fi
