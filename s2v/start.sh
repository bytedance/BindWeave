#!/bin/bash
ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P )"
SCRIPTHOME="$( cd "$(dirname "$0")" ; pwd -P )"

cd $ROOT
git submodule update --init --recursive

rm MindSpeed-MM/pyproject.toml
cp s2v/s2v/pyproject.toml MindSpeed-MM/pyproject.toml

set -x
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip3 uninstall -y mindspeed
cd $ROOT/MindSpeed
pip3 install -r requirements.txt
pip3 install -e .
cd $ROOT/MindSpeed-MM
pip3 install -e .
pip3 install decord==0.6.0
pip3 install mindstudio-probe==8.1.1
pip3 install diffusers==0.33.1
pip3 install ipdb==0.13.13
pip3 install pycocotools==2.0.10

