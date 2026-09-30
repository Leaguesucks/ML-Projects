# Don't forget to grant permission using chmod +x setup.sh
sudo apt update
sudo apt install build-essential
sudo apt install python3-venv
sudo apt install python3-pip
sudo apt install doxygen

python3 -m venv .venv
source .venv/bin/activate
pip install numpy Pillow
