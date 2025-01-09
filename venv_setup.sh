# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Windows activation would be:
# .\venv\Scripts\activate

python3 --version
python3 setup/install_dependencies.py
python3 setup/test_environment.py

