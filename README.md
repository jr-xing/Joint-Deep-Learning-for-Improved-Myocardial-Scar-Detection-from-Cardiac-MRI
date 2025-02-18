# Joint Deep Learning for Improved Myocardial Scar Detection

This repository contains the official implementation of the paper "JOINT DEEP LEARNING FOR IMPROVED MYOCARDIAL SCAR DETECTION FROM CARDIAC MRI".

## Abstract

Automated identification of myocardial scar from late gadolinium enhancement cardiac magnetic resonance images (LGE-CMR) is limited by image noise and artifacts such as those related to motion and partial volume effect. This paper presents a novel joint deep learning (JDL) framework that improves such tasks by utilizing simultaneously learned myocardium segmentations to eliminate negative effects from non-region-of-interest areas. Our method introduces a message passing module where myocardium segmentation information directly guides scar detectors, leading to improved performance compared to state-of-the-art methods.

## Installation

```bash
# Clone this repository
git clone https://github.com/username/repository.git

# Create and activate a new conda environment
conda create -n jdl python=3.8
conda activate jdl

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train the model, run:

```bash
python train.py --config_file configs/config.json
```

Key training parameters can be configured in the JSON config file:
- Network architecture
- Learning rate
- Batch size
- Number of epochs
- Data paths
- Evaluation metrics

### Experiment Tracking

The training process is monitored using Weights & Biases (wandb). Training metrics, validation results, and prediction visualizations are automatically logged.

## Model Architecture

The Joint Deep Learning framework consists of:
- A myocardium segmentation network
- A scar detection network
- A message passing module connecting the two networks

## Results

Our method demonstrates superior performance compared to:
- Two-step segmentation-classification networks
- Multitask learning schemes with indirect interaction

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@inproceedings{xing2023joint,
  title={Joint deep learning for improved myocardial scar detection from cardiac MRI},
  author={Xing, Jiarui and Wang, Shuo and Bilchick, Kenneth C and Patel, Amit R and Zhang, Miaomiao},
  booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## Contact

For questions or issues, please open an issue.
