# DisCoKet

## About

DisCoKet is a toolkit for quantum natural language processing (QNLP).

This project is in early development, so there is no guarantee of
stability.

## Getting started

### Installation

1. Download this repository:
   ```bash
   git clone https://github.com/CQCL/discoket
   ```

2. Enter the repository:
   ```bash
   cd discoket
   ```

#### Automatic Installation

The repository contains a script `install.sh` which, given a directory
name, creates a Python virtual environment and installs DisCoKet.

To install DisCoKet using this script, run:
```bash
bash install.sh <installation-directory>
```

#### Manual Installation

1. DisCoKet has the dependency `depccg` which requires the following
   packages to be installed *before* installing `depccg`:
   ```bash
   pip install cython numpy
   ```
   Further information can be found on the
   [depccg homepage](//github.com/masashi-y/depccg).

2. Install DisCoKet from the local repository using pip:
   ```bash
   pip install .
   ```

3. If using a pretrained depccg parser,
[download a pretrained model](//github.com/masashi-y/depccg#using-a-pretrained-english-parser):
```bash
depccg_en download
```

## Usage

Example - parsing a sentence into a diagram:

```python
from discoket.ccg2diagram import DepCCGParser

depccg_parser = DepCCGParser()
diagram = depccg_parser.sentence2diagram('This is a test sentence')
diagram.draw()
```

Note: all pre-trained depccg models apart from the basic one are broken,
and depccg has not yet been updated to fix this. Therefore, it is
recommended to just use the basic parser, as shown here.

## Testing

Run all tests with the command:

```bash
pytest
```
