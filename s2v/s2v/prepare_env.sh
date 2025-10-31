set -x
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip3 uninstall -y mindspeed
cd /mnt/bn/qdj-hl/qiandongjun/codes/MindSpeed
pip3 install -r requirements.txt
pip3 install -e .
cd /mnt/bn/qdj-hl/qiandongjun/codes/MindSpeed-MM
pip3 install -e .
pip3 install decord==0.6.0
pip3 install mindstudio-probe==8.1.1
pip3 install diffusers==0.33.1
pip3 install ipdb==0.13.13
pip3 install pycocotools==2.0.10
