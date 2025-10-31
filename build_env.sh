set -x
source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip3 install jsonargparse==4.35.0
pip3 install pydantic==2.10.4
pip3 uninstall -y mindspeed

# fetch submodules
git submodule update --init --recursive
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-MM/
cd ..

# install mindspeed
cd MindSpeed
git checkout 6f11a6c9edd409f32a805a71e710b01f9191438f
pip3 install -r requirements.txt
pip3 install -e .
cd ..
cp MindSpeed-MM/examples/qwen2vl/dot_product_attention.py MindSpeed/mindspeed/core/transformer/dot_product_attention.py
# install mindspeed-mm
cd MindSpeed-MM
cp ../pyproject.toml .
pip3 install -e .

cd ..

git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout fa56dcc2a
pip install -e .

pip3 install decord==0.6.0
pip3 install mindstudio-probe
pip3 install diffusers==0.33.1
pip3 install pycocotools==2.0.8
pip install filelock
pip install openai


