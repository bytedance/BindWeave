<<<<<<< HEAD
<h1 align="center">
  BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration
</h1>


<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2510.00438-b31b1b.svg)](https://arxiv.org/pdf/2510.00438)&nbsp;
[![project page](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://lzy-dot.github.io/BindWeave/)&nbsp;
<a href="https://huggingface.co/ByteDance/BindWeave"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>
</div>


 <p align="center">
  <a href="https://arxiv.org/abs/2502.11079"><strong>BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration</strong></a>
</p>

<div align="center">
  <p>
    <a href="https://scholar.google.com/citations?user=WelDcqkAAAAJ&hl=zh-CN">Zhaoyang Li</a><sup> 1,2</sup>, 
    <a href="https://openreview.net/profile?id=~Dongjun_Qian1">Dongjun Qian</a><sup> 2</sup>, 
    <a href="https://scholar.google.com/citations?user=Kp3XAToAAAAJ&hl=zh-CN">Kai Su</a><sup> 2*</sup>, 
    <a href="https://scholar.google.com/citations?user=G6xrfhYAAAAJ&hl=zh-CN">Qishuai Diao</a><sup> 2</sup>, 
    <a href="https://openreview.net/profile?id=~Xiangyang_Xia1">Xiangyang Xia</a><sup> 2</sup>, 
    <a href="https://openreview.net/profile?id=~Chang_Liu71">Chang Liu</a><sup> 2</sup>,
    <a href="https://scholar.google.com/citations?user=rtO5VmQAAAAJ&hl=zh-CN">Wenfei Yang</a><sup> 1</sup>, 
    <a href="https://scholar.google.com/citations?user=9sCGe-gAAAAJ&hl=en">Tianzhu Zhang</a><sup> 1*</sup>, 
    <a href="https://shallowyuan.github.io/">Zehuan Yuan</a><sup> 2</sup>
  </p>
  <p>
    <small>
      <sup>1</sup>University of Science and Technology of China <sup>2</sup>ByteDance
      <br>
      <sup>*</sup>Corresponding Author
    </small>
  </p>
</div>


<p align="center">
<img src="assets/figure1.png" width=95%>
<p>

## 🗓️ Todo List

- [x] Release inference code
- [x] Release checkpoint of BindWeave_Wan_14B
- [x] Release training code of BindWeave


## 📖 Overview
BindWeave is a unified subject-consistent video generation framework for single- and multi-subject prompts, built on an MLLM-DiT architecture that couples a pretrained multimodal large language model with a diffusion transformer.
It achieves cross-modal integration via entity grounding and representation alignment, leveraging the MLLM to parse complex prompts and produce subject-aware hidden states that condition the DiT for high-fidelity generation.

<!-- ## Project Workflow and Branching
This project is organized into a multi-branch structure to ensure that each major workflow—feature extraction, training, and inference—is independent and self-contained. This design prevents conflicts between dependencies and submodule versions, reduces memory overhead, and creates a clean and stable environment for each task.
The `main` branch serves as the main entry point and contains this primary README. To begin working, please switch to the branch that corresponds to your goal.
| Branch                 | Purpose                                                                | Command to Switch          |
| ---------------------- | ---------------------------------------------------------------------- | -------------------------- |
| `feature_extraction`   | Pre-computing features required for training and inference.            | `git switch feature_extraction` |
| `train`                | Training the BindWeave model from scratch.                             | `git switch train`         |
 
After switching, please consult the `README.md` file within that branch for detailed instructions. -->

## 🎥 Demo
![BindWeave Video Generation Demo](assets/bindweave_demo.gif)


<!-- # BindWeave Inference Process

This document outlines the specific inference workflow for the `infer` branch. To optimize memory usage, this process adopts an offline approach by pre-extracting and saving hidden states from the Qwen2.5-VL model before the main inference stage.

The process is broken down into three main steps:

1.  **Prompt Refinement**: Enhance the initial prompt to more accurately describe the subject in the reference image.
2.  **Hidden States Extraction**: Extract and save the hidden states from Qwen2.5-VL.
3.  **Inference**: Run the final inference, loading the pre-computed hidden states as part of the input. -->

## ⚡️ Quickstart

### Installation
Clone the repo:
```sh
git clone 
cd BindWeave
```

Install dependencies:
```bash
bash build_env.sh
```

## Step 1: Prompt Refinement (Optional)

Before extracting features, you can first refine the original prompt. This step leverages a MLLM to generate a more detailed and descriptive prompt that better captures the key attributes of the subject in the reference image.

**Usage:**

```bash
bash scripts/prompt_refine.sh
```

## Step 2: Hidden States Extraction

> **Important Note on Workflow Independence**
>
> To ensure each workflow (feature extraction, training, inference) remains independent and to avoid conflicts from differing submodule versions, the necessary scripts for this step reside in the `feature_extraction` branch.
>
> We will temporarily switch to that branch to generate the features, and then return to the `infer` branch to use them. 

Follow these steps to generate the hidden states:

1.  **Switch to the `feature_extraction` branch:**
    This branch contains the specialized scripts for feature generation.
    ```bash
    git switch feature_extraction
    ```

2.  **Run the extraction script:**
    Execute the script to process your inference samples and save the hidden state files.
    ```bash
    bash scripts/hiddenstates_extraction.sh
    ```

3.  **Return to the `infer` branch:**
    Once the extraction is complete, switch back to your original branch to proceed with inference. The generated files will remain available.
    ```bash
    git switch infer
    ```

Now you have the required hidden state files and are ready for the next step in the `infer` workflow.


## Step 3: Inference
<!-- Before running the inference code, you need to download the 14B original model of Wan2.1, as our BindWeave depends on the Wan2.1 VAE and text encoder:
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers --local-dir ./pretrained_model/wanx/Wan2.1-I2V-14B-720P-Diffusers
``` -->
Before running the inference code, you need to download the original 14B model of WanX 2.1. This is crucial because BindWeave depends on its components like the VAE and text encoder.

1.  **Download the Pre-trained Model:**
    First, use the Hugging Face CLI to download the model weights. The command below will place them in the `./pretrained_model/wanx/` directory.
    ```bash
    pip install "huggingface_hub[cli]"
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers --local-dir ./pretrained_model/wanx/Wan2.1-I2V-14B-720P-Diffusers
    ```

2.  **Update the Configuration File:**
    After the download is complete, you must update the configuration file at `configs/inference/inference_model_s2v.json`. Ensure that the paths for the following components correctly point to the directories you just downloaded:
    *   `vae`
    *   `tokenizer`
    *   `text_encoder`
    *   `image_encoder`

Then download the BindWeave model:
```bash
huggingface-cli download ByteDance/BindWeave --local-dir ./BindWeave_14B
```

#### Weight Conversion
After downloading the BindWeave model, you need to convert the transformer weights to the MM format. Run the conversion script as follows:
```
python convert_ckpt.py \
  --source_path ./BindWeave_14B/ \
  --target_path ./BindWeave_14B/ \
  --mode convert_to_mm
```


Run Subject-to-Video Generation
```bash
bash script/inference_s2v.sh
```
You can modify the corresponding paths in `'BindWeave/configs/inference/inference_model_s2v.json'`, where:
- `BASE_IMG_DIR`: Root directory of the reference images.
- `META_PATH`: Sample JSON file used during inference.
- `OUT_DIR`: Output directory for inference results.



Using the provided sample cases (i.e., the default path configuration), running `bash script/inference_s2v.sh` will produce the following generated results:

<table style="width: 100%; border-collapse: collapse; text-align: center; border: 1px solid #ccc;">
  <tr>
    <th style="text-align: center;">
      <strong>Reference Images</strong>
    </th>
    <th style="text-align: center;">
      <strong>Generated Videos (720P)</strong>
    </th>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="s2v/OpenS2V-Eval/Images/human.jpg" alt="Image 1" style="height: 100px;">
      <img src="s2v/OpenS2V-Eval/Images/dog.jpg" alt="Image 2" style="height: 100px;">
      <img src="s2v/OpenS2V-Eval/Images/school.jpg" alt="Image 2" style="height: 100px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <img src="s2v/OpenS2V-Eval/Results/faceobj.gif" alt="GIF 1" style="width: 400px;">
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="s2v/OpenS2V-Eval/Images/man.png" alt="Image 1" style="height: 120px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <img src="s2v/OpenS2V-Eval/Results/singleface.gif" alt="GIF 1" style="width: 400px;">
    </td>
  </tr>
  
  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="s2v/OpenS2V-Eval/Images/woman1.jpg" alt="Image 1" style="height: 150px;">
      <img src="s2v/OpenS2V-Eval/Images/woman2.jpg" alt="Image 1" style="height: 150px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <img src="s2v/OpenS2V-Eval/Results/multihuman.gif" alt="GIF 1" style="width: 400px;">
    </td>
  </tr>

</table>

> The GIF videos are compressed.

##  OpenS2V-Eval Performance 🏆
BindWeave achieves a solid score of 57.61 on the [OpenS2V-Eval](https://huggingface.co/spaces/BestWishYsh/OpenS2V-Eval) benchmark, highlighting its robust capabilities across multiple evaluation dimensions and demonstrating competitive performance against several leading open-source and commercial systems.

| Model | TotalScore↑ | AestheticScore↑ | MotionSmoothness↑ | MotionAmplitude↑ | FaceSim↑ | GmeScore↑ | NexusScore↑ | NaturalScore↑ |
|------|----|----|----|----|----|----|----|----|
| [BindWeave](https://lzy-dot.github.io/BindWeave/) | 57.61% | 45.55% | 95.90% | 13.91% | 53.71% | 67.79% | 46.84% | 66.85% |
| [VACE-14B](https://github.com/ali-vilab/VACE) | 57.55% | 47.21% | 94.97% | 15.02% | 55.09% | 67.27% | 44.08% | 67.04% |
| [Phantom-14B](https://github.com/Phantom-video/Phantom) | 56.77% | 46.39% | 96.31% | 33.42% | 51.46% | 70.65% | 37.43% | 69.35% |
| [Kling1.6(20250503)](https://app.klingai.com/cn/) | 56.23% | 44.59% | 86.93% | 41.6% | 40.1% | 66.2% | 45.89% | 74.59% |
| [Phantom-1.3B](https://github.com/Phantom-video/Phantom) | 54.89% | 46.67% | 93.3% | 14.29% | 48.56% | 69.43% | 42.48% | 62.5% |
| [MAGREF-480P](https://github.com/MAGREF-Video/MAGREF) | 52.51% | 45.02% | 93.17% | 21.81% | 30.83% | 70.47% | 43.04% | 66.9% |
| [SkyReels-A2-P14B](https://github.com/SkyworkAI/SkyReels-A2) | 52.25% | 39.41% | 87.93% | 25.6% | 45.95% | 64.54% | 43.75% | 60.32% |
| [Vidu2.0(20250503)](https://www.vidu.cn/) | 51.95% | 41.48% | 90.45% | 13.52% | 35.11% | 67.57% | 43.37% | 65.88% |
| [Pika2.1(20250503)](https://pika.art/) | 51.88% | 46.88% | 87.06% | 24.71% | 30.38% | 69.19% | 45.4% | 63.32% |
| [VACE-1.3B](https://github.com/ali-vilab/VACE) | 49.89% | 48.24% | 97.2% | 18.83% | 20.57% | 71.26% | 37.91% | 65.46% |
| [VACE-P1.3B](https://github.com/ali-vilab/VACE) | 48.98% | 47.34% | 96.8% | 12.03% | 16.59% | 71.38% | 40.19% | 64.31% |


### BibTeX
```bibtex
@article{li2025bindweave,
  title={BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration},
  author={Li, Zhaoyang and Qian, Dongjun and Su, Kai and Diao, Qishuai and Xia, Xiangyang and Liu, Chang and Yang, Wenfei and Zhang, Tianzhu and Yuan, Zehuan},
  journal={arXiv preprint arXiv:2510.00438},
  year={2025}
}
