#### Train Branch Overview
BindWeave’s training workflow is isolated in the `train` branch to keep dependencies, submodules, and runtime environments independent from feature extraction and inference. This separation avoids version conflicts and reduces memory overhead by loading only what’s needed for training.


Before running the training code, first extract features using the `feature_extraction` branch and download the pre-trained Wanx 2.1 model, as BindWeave is fine-tuned from it.
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers --local-dir ./pretrained_model/wanx/Wan2.1-I2V-14B-720P-Diffusers
```

Then, perform weight conversion for the Transformer component of the downloaded Wan 2.1 model. For details, please refer to the MindSpeed-MM Wan 2.1 examples (v2.1.0): https://gitee.com/ascend/MindSpeed-MM/tree/2.1.0/examples/wan2.1.

#### Start Training
Ensure you are on the `train` branch and have completed the prerequisites:
- Features extracted via the `feature_extraction` branch.
- Pre-trained Wanx 2.1 (14B) model downloaded and transformed.
##### 
Open the training script to verify and adjust paths and hyperparameters:
```bash
bash s2v/s2v/examples/wan2.1/14b/s2v/train_s2v.sh
```
