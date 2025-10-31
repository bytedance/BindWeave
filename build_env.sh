set -x
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip3 install jsonargparse==4.35.0
pip3 install pydantic==2.10.4
pip3 uninstall -y mindspeed

# fetch submodules
git submodule update --init --recursive

cp -r Megatron-LM/megatron MindSpeed-MM/

# install mindspeed
cd MindSpeed
pip3 install -r requirements.txt
pip3 install -e .

# install mindspeed-mm
cd ../MindSpeed-MM
cp ../pyproject.toml .
pip3 install -e .
pip3 install decord==0.6.0
pip3 install mindstudio-probe
pip3 install diffusers==0.33.1
pip3 install pycocotools==2.0.8
pip install openai
