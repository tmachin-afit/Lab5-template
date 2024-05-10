## RUNNING ON ACE
# ensure updated system
sudo apt update && sudo apt upgrade -y
apt-get install libsm6 \
    libxext6
snap refresh
snap install --classic code
# install pip (python package manager) if not installed already
sudo apt install -y python3-tk
python3 -m pip install --upgrade pip
# install python requirements
python3 -m pip install -r requirements.txt
