# pysptk

[![PyPI](https://img.shields.io/pypi/v/pysptk.svg)](https://pypi.org/project/pysptk/)
[![Python package](https://github.com/r9y9/pysptk/actions/workflows/ci.yaml/badge.svg)](https://github.com/r9y9/pysptk/actions/workflows/ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/pysptk/badge/?version=latest)](https://readthedocs.org/projects/pysptk/?badge=latest)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![][docs-stable-img]][docs-stable-url]
[![][docs-latest-img]][docs-latest-url]
[![DOI](https://zenodo.org/badge/41618046.svg)](https://zenodo.org/badge/latestdoi/41618046)

A python wrapper for [Speech Signal Processing Toolkit (SPTK)](http://sp-tk.sourceforge.net/).

**NOTE**: pysptk is based on a modified version of SPTK ([r9y9/SPTK](https://github.com/r9y9/SPTK)).

## Documentation

- [**STABLE**][docs-stable-url] &mdash; **most recently tagged version of the documentation.**
- [**LATEST**][docs-latest-url] &mdash; *in-development version of the documentation.*

## Demonstration notebook

- [Introduction notebook](https://nbviewer.org/github/r9y9/pysptk/blob/master/examples/pysptk%20introduction.ipynb): a brief introduction to pysptk
- [Speech analysis and re-synthesis](https://nbviewer.org/github/r9y9/pysptk/blob/master/examples/Speech%20analysis%20and%20re-synthesis.ipynb): a demonstration notebook for speech analysis and re-synthesis. Synthesized audio examples (English) are available on the notebook.

## Installation

Install the latest release from PyPI.

```bash
pip install pysptk
```

Install the current development version directly from GitHub.

```bash
pip install git+https://github.com/r9y9/pysptk
```

Source builds require a C compiler. For a local checkout, clone the repository
with `git clone --recurse-submodules https://github.com/r9y9/pysptk.git` and
run `python -m pip install .` from the checkout.

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://pysptk.readthedocs.io/en/latest/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://pysptk.readthedocs.io/en/stable/
