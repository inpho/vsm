#VSM

[![Build Status](https://img.shields.io/travis/inpho/vsm.svg?label=UNIX+build)](https://travis-ci.org/inpho/vsm)
[![Appveyor](https://img.shields.io/appveyor/ci/JaimieMurdock/vsm.svg?label=Windows+build)](https://ci.appveyor.com/project/JaimieMurdock/vsm)
[![Coveralls](https://img.shields.io/coveralls/inpho/vsm.svg)](https://coveralls.io/r/inpho/vsm)
[![GitHub license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/inpho/vsm/blob/master/LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/vsm.svg)](https://pypi.python.org/pypi/vsm)

**Note:** More than likely, you are looking for the [InPhO Topic Explorer](http://github.com/inpho/topic-explorer). This library is for the machine learning implementations underlying the Topic Explorer and is updated much slower than the user interfaces. If you are using the Topic Explorer, please file issues there and the developers will triage appropriately.

Vector Space Model Framework developed for the 
[InPhO Project](https://inpho.cogs.indiana.edu/).

Documentation can be found in the module and [here](http://inpho.github.io/vsm/).

##Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage its environment and
build the Cython extension. Install `uv` first (see the
[installation instructions](https://docs.astral.sh/uv/getting-started/installation/)),
then:

```
git clone git@github.com:inpho/vsm.git
cd vsm
uv sync --group dev
```

`uv sync` creates a `.venv`, resolves the dependencies, and installs `vsm` as an
editable package (compiling the Cython extension in place). Python 3.10 or newer
is required; pass `--python 3.12` to select a specific interpreter.

To run the tests:

```
uv run pytest unit_tests/*
```

The trailing glob is needed because the test modules are named `tests_*.py`,
which does not match pytest's default discovery pattern.

###Installing without uv

The traditional workflow still works; `[build-system]` in `pyproject.toml` pulls
in Cython and NumPy for the build, so no separate pre-install step is needed:

```
git clone git@github.com:inpho/vsm.git
cd vsm
pip install -e .
```
