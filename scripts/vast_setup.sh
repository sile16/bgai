#!/bin/bash

git clone https://github.com/sile16/bgai.git
cd bgai
python -m venv venv-bgai
source venv-bgai/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=venv-bgai --display-name "Python (bgai)"
