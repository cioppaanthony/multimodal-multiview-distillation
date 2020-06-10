#!/bin/bash
set -euf -o pipefail

echo --------------------------------
echo APT-GET
echo --------------------------------
apt-get -y update
apt-get -y upgrade
apt-get -y install htop

echo --------------------------------
echo APT-GET PYTHON
echo --------------------------------
apt-get -y install python-pip
apt-get -y install python3-pip python3-dev

echo --------------------------------
echo PIP INSTALL
echo --------------------------------
python3 -m pip install --upgrade pip
python3 -m pip install h5py==2.10.0
python3 -m pip install matplotlib==3.1.2
python3 -m pip install numpy==1.14.0
python3 -m pip install opencv-python-headless==4.1.2.30
python3 -m pip install opencv-contrib-python-headless==4.1.2.30
python3 -m pip install Pillow==5.0.0
python3 -m pip install scipy==1.0.1
python3 -m pip install torch==1.0.1.post2 
python3 -m pip install torchvision==0.2.0
python3 -m pip install tqdm==4.19.4
apt-get -y install python3-tk


apt-get -y install libglib2.0-0
