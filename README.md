# DACESR

### [Paper]()

> **EDACESR: Degradation-Aware Conditional Embedding for Real-World Image Super-Resolution** <br>
> [Xiaoyan Lei](https://scholar.google.com/citations?hl=zh-CN&user=o8GJ_YMAAAAJ/), [Wenlong Zhang](https://wenlongzhang0517.github.io/), [Hui Liang](), [Weifeng Cao]() and [Qiuting Lin](). <br>
> In TIP.

### Abstract

Multimodal large models have shown excellent ability in addressing image super-resolution in real-world scenarios by leveraging language class as condition information, yet their abilities in degraded images remain limited. In this paper, we first revisit the capabilities of the Recognize Anything Model (RAM) for degraded images by calculating text similarity. We find that directly using contrastive learning to fine-tune RAM in the degraded space is difficult to achieve acceptable results. To address this issue, we employ a degradation selection strategy to propose a Real Embedding Extractor (REE), which achieves significant recognition performance gain on degraded image content through contrastive learning. Furthermore, we use a Conditional Feature Modulator (CFM) to incorporate the high-level information of REE for a powerful Mamba-based network, which can leverage effective pixel information to restore image textures and produce visually pleasing results. Extensive experiments demonstrate that the REE can effectively help image super-resolution networks balance fidelity and perceptual quality, highlighting the great potential of Mamba in real-world applications. 

Overall pipeline of the DACESR:

![illustration](Pipeline.pdf)

For more details, please refer to our paper.

#### Getting started

- Clone this repo.
```bash
git clone https://github.com/csjliang/DACESR
cd DACESR
```

- Install dependencies. (Python 3 + NVIDIA GPU + CUDA. Recommend to use Anaconda)
```bash
pip install -r requirements.txt
```

- Prepare the training and testing dataset by following this [instruction](datasets/README.md).
- Prepare the pre-trained models by following this [instruction](experiments/README.md).

#### Training

First, check and adapt the yml file ```options/train/DACESR/train_DACESR.yml```, then

- Single GPU:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python dacesr/train.py -opt options/train/DACESR/train_DACESR.yml --auto_resume
```

- Distributed Training:
```bash
YTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4335 dacesr/train.py -opt options/train/DACESR/train_DACESR.yml --launcher pytorch --auto_resume

```

Training files (logs, models, training states and visualizations) will be saved in the directory ```./experiments/{name}```

#### Testing

First, check and adapt the yml file ```options/test/DACESR/test_DACESR.yml```, then run:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/DACESR/test_DACESR.yml
```

Evaluating files (logs and visualizations) will be saved in the directory ```./results/{name}```

### License

This project is released under the Apache 2.0 license.

### Citation
```
@inproceedings{jie2022DASR,
  title={Efficient and Degradation-Adaptive Network for Real-World Image Super-Resolution},
  author={Liang, Jie and Zeng, Hui and Zhang, Lei},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

### Acknowledgement
This project is built based on the excellent [BasicSR](https://github.com/xinntao/BasicSR) project.

### Contact
Should you have any questions, please contact me via `xyan_lei@163.com`.
