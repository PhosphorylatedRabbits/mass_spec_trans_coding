#!/usr/bin/env bash

# build sphinx module rst files from docstrings
sphinx-apidoc --ext-githubpages --ext-autodoc --ext-todo --ext-coverage --ext-viewcode -f -o _modules ../mstc ../*test_*
