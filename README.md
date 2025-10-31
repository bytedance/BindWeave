### Branch Overview
This branch is dedicated to feature extraction for the BindWeave project.
The primary goal is to pre-compute and cache essential features, which significantly speeds up both the training and inference cycles. By decoupling feature extraction from the main model's logic, we can iterate faster and manage data more efficiently.
This branch introduces two main workflows:
1. Training Feature Extraction: Processes an entire dataset to extract and save features required for training the BindWeave model.
2. Inference Feature Extraction: Extracts deep features (hidden states) from a MLLM, Qwen2.5-VL, to be used as conditioning during inference.
Note: This is a development branch. The scripts and workflows here are designed to prepare data for use in other branches (like `train` or `infer`).

###  1. Installation
First, clone the repository to your machine, and then switch to the `feature_extraction` branch.
```bash
# 1. Clone the repository
git clone
# 2. Navigate into the repository directory
cd BindWeave
# 3. Switch to the feature_extraction branch
git switch feature_extraction

bash build_env.sh
```
To extract the necessary features, you must first download Qwen2.5-VL (7B) to serve as the source for hidden states, and Wanx 2.1 (14B) to provide the text encoder for extracting text embeddings and the core VAE.

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./pretrained_model/Qwen2.5-VL-7B-Instruct
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers --local-dir ./pretrained_model/wanx/Wan2.1-I2V-14B-720P-Diffusers
```

### 2. Weight Conversion (`hf2mm`)
The `mm-convert` utility is required to adapt pre-trained weights for MindSpeed-MM, which uses customized layer names.
**Key Features:**
-   **Bidirectional Conversion:** Convert weights between the standard Hugging Face format and the MindSpeed-MM format.
-   **Re-sharding for Parallelism:** Re-split weights for Pipeline Parallel (PP) configurations.
```
cd MindSpeed-MM/examples/qwen2.5vl

mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "pretrained_model/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "pretrained_model/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[28]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 1
```

For details, please refer to the following MindSpeed-MM example directories:
- Qwen2.5-VL: https://gitee.com/ascend/MindSpeed-MM/tree/2.1.0/examples/qwen2.5vl
- WAN 2.1: https://gitee.com/ascend/MindSpeed-MM/tree/2.1.0/examples/wan2.1

### 3. Extract Training and Inference Features
Prepare the required features for both training and inference:
- Training features: features used during the training phase.
- Inference features: features used for the final inference step.

#### 3.1 Training Data Preparation
Organize your dataset in the following structure:
```
</dataset>
  ├── data.json
  ├── videos
  │  ├── video0001.mp4
  │  ├── video0002.mp4
```
- The `videos/` directory stores all video files.
- The `data.json` file contains all video–text pair information. Example:
```
[
  {
    "path": "videos/video0001.mp4",
    "cap": "Video caption1.",
    "num_frames": 81,
    "fps": 24,
    "resolution": { "height": 720, "width": 1280 }
  },
  {
    "path": "videos/video0002.mp4",
    "cap": "Video caption2.",
    "num_frames": 81,
    "fps": 24,
    "resolution": { "height": 720, "width": 1280 }
  }
  // ...
]
```
Update `configs/feature_extract/data.txt`:
- Each line represents one dataset.
- Format: `<dataset_folder_path>,<data_json_path>`
Example:
/path/to/dataset,/path/to/dataset/data.json

##### 3.1 Extract Training Features
Run the training feature extraction script:
```
bash scripts/feature_extraction.sh
```
##### 3.2 Extract Inference Features (Hidden States)
Run the hidden states extraction script:
```
bash scripts/hiddenstates_extraction.sh
```



