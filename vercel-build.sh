#!/bin/bash
# Install Python 3.9
pyenv install 3.9.18 --skip-existing
pyenv global 3.9.18

# Install dependencies
pip install -r requirements-vercel.txt
