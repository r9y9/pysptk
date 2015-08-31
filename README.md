# pysptk

[![Build Status](https://travis-ci.org/r9y9/pysptk.svg?branch=master)](https://travis-ci.org/r9y9/pysptk)
[![Coverage Status](https://coveralls.io/repos/r9y9/pysptk/badge.svg?branch=master&service=github)](https://coveralls.io/github/r9y9/pysptk?branch=master)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

pysptk is a python wrapper for the [Speech Signal Processing Toolkit (SPTK)](http://sp-tk.sourceforge.net/), which provides a lot of functionalities for speech signal processing such as linear prediction analysis, mel-cepstrum analysis, generalized cepstrum analysis and mel-generalized cepstrum analysis to name a few. See the original project page for more details.


**NOTE**: pysptk is based on a modified version of SPTK ([r9y9/SPTK](https://github.com/r9y9/SPTK)).

## Documentation

A reference manual of the SPTK can be found at http://sp-tk.sourceforge.net/.

## Demonstration notebook

-  [Introduction notebook](http://nbviewer.ipython.org/github/r9y9/pysptk/blob/master/examples/pysptk%20introduction.ipynb): a brief introduction to pysptk

## Installation

```bash
git submodule update --init --recursive
pip install -e .
```
