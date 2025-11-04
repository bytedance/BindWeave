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

## Project Workflow and Branching
This project is organized into a multi-branch structure to ensure that each major workflow—feature extraction, training, and inference—is independent and self-contained. This design prevents conflicts between dependencies and submodule versions, reduces memory overhead, and creates a clean and stable environment for each task.
The `main` branch serves as the main entry point and contains this primary README. To begin working, please switch to the branch that corresponds to your goal.
| Branch                 | Purpose                                                                | Command to Switch          |
| ---------------------- | ---------------------------------------------------------------------- | -------------------------- |
| `feature_extraction`   | Pre-computing features required for training and inference.            | `git switch feature_extraction` |
| `train`                | Training the BindWeave model from scratch.                             | `git switch train`         |
| `infer`                | Running inference with a pre-trained model.                            | `git switch infer`         |
After switching, please consult the `README.md` file within that branch for detailed instructions.

###  OpenS2V-Eval Performance 🏆
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