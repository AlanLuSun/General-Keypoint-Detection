# General-Keypoint-Detection
## 1. Introduction
- This is the source code implementation for the ECCV 2026 paper [GKDT: General Keypoint Detection Transformer](https://arxiv.org/pdf/2607.00752)
- The major contributions of this work are in two aspects: 1) We present ***a powerful yet highly practical foundation model, called GKDT, to handle general keypoint detection***. GKDT supports visual prompt, text prompt, and both, enabling few-shot, zero-shot, and multimodal prompted detection. GKDT is open-world, thus it also supports continual learning. To our best knowledge, GKDT is the first DINOv3 based model for general keypoint detection (GKD); 2) We present ***a large-scale high-quality dataset MegaKPT that unifies 29 public datasets with over 1.3 million object instances*** for the research of general keypoint detection.
- The source codes, models, and dataset is free for academic research, while commercial use is prohibited. Please cite our paper if you find them helpful, thank you!

<!-- # We are working on releasing the dataset, models, and codes. We will release them within 1~2 weeks. Thanks for your patience and stay tuned! (2026.07.15) -->


## 2. Released GKDT Models
We release two sets of GKDT models. One is trained using the entire MegaKPT dataset (including training, validation and/or test sets of each component dataset) to pursue great performance for real-world testing and applications; Another set is only trained by the combination of training sets of each component dataset for the purpose of research. The model variants GKDT-L (default model in our paper) and GKDT-H use the DINOv3-L and DINOv3-H visual backbones, respectively.
 
### 2.1 For real-world testing & Application
Results across 22 datasets' test sets with 1000 episodes and metric PCK@0.1:
|        |Prompt|Animal|AwA  |CUB  |NABird|AP-10K|Vinegar fly|Locust|Mourse5k|Macaque|Tiger|Animal Kin.|COCO val|HumanArt|300W |Ani.Web|OneHand|HInt |Keypoint-5|CarFusion|D.Fashion2|Cephalo|Hand X-ray|
|--------|:----:|:----:|:---:|-----|:----:|:----:|:---------:|:----:|:------:|:-----:|:---:|:---------:|:------:|:------:|:---:|:-----:|:-----:|:---:|:--------:|:-------:|:--------:|:-----:|:--------:|
|GKDT-L  |visual|89.13 |94.84|98.79|97.99 |95.52 |99.42      |99.32 |98.46   |91.45  |95.86|96.70      |90.59   |88.93   |98.52|95.28  |97.77  |88.85|97.46     |97.51    |91.19     |99.98	 |100.00    |
|        |text  |94.02 |97.08|99.34|98.27 |96.78 |99.89      |99.80 |98.55   |95.05  |97.75|97.74      |94.46   |94.32   |99.76|97.47  |98.90  |94.26|98.62     |98.13    |95.31     |99.98	 |100.00    |
|        |both  |93.12 |96.80|99.28|98.24 |96.50 |99.86      |99.74 |98.56   |94.62  |97.56|97.62      |94.51   |93.70   |99.68|97.42  |98.73  |93.45|98.42     |97.91    |94.64     |99.98	 |100.00    |
|GKDT-H  |visual|92.10 |95.69|99.22|98.85 |97.32 |99.56      |99.33 |98.72   |91.97  |97.03|98.42      |92.33   |91.20   |99.13|96.79  |98.39  |93.63|98.52     |98.17    |91.24     |100.00 |100.00    |
|        |text  |95.48 |98.49|99.56|99.04 |98.29 |99.95      |99.74 |98.76   |95.88  |98.34|98.96      |95.56   |96.03   |99.93|98.81  |99.36  |97.05|99.16     |98.80    |95.56     |100.00 |100.00    |
|        |both  |94.73 |98.21|99.52|99.04 |98.09 |99.94      |99.67 |98.74   |95.54  |98.20|98.90      |95.54   |95.59   |99.85|98.77  |99.25  |96.54|99.12     |98.58    |94.54     |100.00 |100.00    |

Download for applications: [GKDT-L](https://huggingface.co/changshenglu/GKDT-L_for_App/tree/main) | [GKDT-H](https://huggingface.co/changshenglu/GKDT-H_for_App/tree/main)

### 2.2 For research
Results across 22 datasets' test sets with 1000 episodes and metric PCK@0.1:
|        |Prompt|Animal|AwA  |CUB  |NABird|AP-10K|Vinegar fly|Locust|Mourse5k|Macaque|Tiger|Animal Kin.|COCO val|HumanArt|300W |Ani.Web|OneHand|HInt |Keypoint-5|CarFusion|D.Fashion2|Cephalo|Hand X-ray|
|--------|:----:|:----:|:---:|-----|:----:|:----:|:---------:|:----:|:------:|:-----:|:---:|:---------:|:------:|:------:|:---:|:-----:|:-----:|:---:|:--------:|:-------:|:--------:|:-----:|:--------:|
|GKDT-L  |visual|81.11 |84.64|97.71|96.25 |89.25 |97.59      |98.66 |96.40   |88.82  |92.33|90.74      |88.64   |82.92   |96.99|82.54  |92.56  |69.79|83.01     |94.15    |90.96     |99.46  |99.28     |
|        |text  |73.27 |92.80|98.58|96.94 |91.93 |98.91      |99.49 |97.37   |93.75  |95.88|93.77      |93.59   |90.48   |99.04|86.36  |95.36  |82.64|88.15     |96.25    |95.34     |99.49  |99.62     |
|        |both  |76.87 |91.14|98.48|96.86 |91.50 |98.79      |99.37 |97.29   |93.15  |95.40|93.29      |93.52   |89.68   |98.87|86.24  |95.12  |81.13|87.04     |95.92    |94.58     |99.49  |99.60     |
|GKDT-H  |visual|81.73 |87.26|97.89|96.79 |90.16 |97.88	     |98.69 |96.38   |89.25  |92.83|92.45      |90.14   |84.33   |97.36|83.48  |94.00  |70.54|82.69     |95.37    |90.97     |99.63  |99.86     |
|        |text  |74.01 |93.91|98.83|97.20 |92.53 |99.11      |99.49 |96.68   |94.09  |96.32|95.13      |94.15   |91.51   |99.35|87.63  |96.41  |83.47|88.50     |96.90    |95.77     |99.73  |99.90     |
|        |both  |76.62 |92.57|98.73|97.14 |92.02 |99.01      |99.38 |96.61   |93.52  |95.85|94.69      |94.18   |90.81   |99.12|87.43  |96.14  |82.16|87.51     |96.51    |94.96     |99.72  |99.92     |

Download for research: [GKDT-L](https://huggingface.co/changshenglu/GKDT-L_for_Research) | [GKDT-H](https://huggingface.co/changshenglu/GKDT-H_for_Research)


## 3. Environment Setup





## 4. Real-World Testing & Applications





## 5. MegaKPT Dataset






## 6. Model Training and Evaluation






## 7. Citation
```
@inproceedings{lu2026gkdt,
  title={GKDT: General Keypoint Detection Transformer},
  author={Lu, Changsheng and Chen, Yuxin and Gui, Haokun and Wang, Rong and Yang, Jie and Yang, Harry and Hengel, Anton van den and Jia, Jiaya},
  booktitle={European Conference on Computer Vision},
  year={2026},
  organization={Springer}
}
@inproceedings{lu2024openkd,
  title={Openkd: Opening prompt diversity for zero-and few-shot keypoint detection},
  author={Lu, Changsheng and Liu, Zheyuan and Koniusz, Piotr},
  booktitle={European Conference on Computer Vision},
  pages={148--165},
  year={2024},
  organization={Springer}
}
@inproceedings{lu2022few,
  title={Few-shot keypoint detection with uncertainty learning for unseen species},
  author={Lu, Changsheng and Koniusz, Piotr},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={19394--19404},
  year={2022},
  organization={IEEE}
}
```
