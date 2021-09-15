#!/usr/bin/env bash

GLOBAL_FLAG='--global'
if [ "$#" -ne 1 ]; then
    echo "Lambeq installer.

Usage:
  install.sh $GLOBAL_FLAG
  install.sh <virtual_environment>"
    exit 1
fi

VENV=$1
if [ $VENV != $GLOBAL_FLAG ]; then
    if [ -d $VENV ]; then
        echo -n "'$VENV' exists. Use this as virtual environment? [y/N] "
        read answer
        if [ "$answer" == "${answer#[Yy]}" ]; then exit; fi
    else
        echo "Creating virtual environment at '$VENV'..."
        python3 -m venv $VENV
    fi
    PYTHON=$VENV/bin/python
else
    PYTHON=python3
fi

echo 'Installing dependencies...'
$PYTHON -m pip install --upgrade pip wheel
$PYTHON -m pip install cython numpy

echo 'Installing Lambeq...'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
$PYTHON -m pip install --use-feature=in-tree-build "$SCRIPT_DIR"

MODEL_DIR="$($PYTHON -c 'from depccg.download import MODEL_DIRECTORY, MODELS; print(MODEL_DIRECTORY / MODELS["en"][1])')"
if [ ! -d "$MODEL_DIR" ]; then
    echo -n 'Download pre-trained depccg parser? [Y/n] '
    read answer
    if [ -n "$answer" ] && [ "$answer" == "${answer#[Yy]}" ]; then exit; fi

    echo 'Downloading parser...'
    $PYTHON -m depccg en download
fi
