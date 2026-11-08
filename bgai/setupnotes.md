
#colab - original turbozero setup
%pip uninstall jax jaxlib jax[cuda] jax_cuda12_plugin jax_cuda12_pjrt

%pip install git+https://github.com/lowrollr/turbozero.git@main jax[cuda]
%pip install git+https://github.com/sile16/pgx.git@master


# vast.ai
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git

curl https://pyenv.run | bash

pyenv install 3.12

curl -fsSL https://tailscale.com/install.sh | sh

git clone https://github.com/sile16/bgai.git
cd bgai
python -m venv venv-bgai
source venv-bgai/bin/activate
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=venv-bgai --display-name "Python (bgai)"
