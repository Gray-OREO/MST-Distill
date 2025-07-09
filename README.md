# MST-Distill

**Paper**: [MST-Distill: Mixture of Specialized Teachers for Cross-Modal Knowledge Distillation]()

**Authors**: Hui Li, Pengfei Yang, Juanyang Chen, Le Dong, Yanxin Chen, Quan Wang

**Published in**: ACM Multimedia 2025

---

## 🎯 Overview

MST-Distill addresses the key challenges of path selection and knowledge drift in cross-modal knowledge distillation by constructing diverse teacher model ensembles, implementing instance-level dynamic distillation through routing networks, and employing feature masking networks to suppress modality discrepancies, significantly improving knowledge transfer quality across different modalities.

![Overall model architecture of ADCMT.](ims/Figs_framework.jpg)

---

## 📋 Requirements

**Main Dependencies:**

- Python >= 3.9
- PyTorch >= 2.1

All experiments are conducted on a server equipped with an Intel Xeon Gold 6248R CPU and an NVIDIA A100 GPU.

Noting that the results may be still not the same among different implement devices. See [randomness@Pytorch Docs](https://pytorch.org/docs/stable/notes/randomness.html).

---

## 🚀 Quick Start

### 1. Dataset Preparation

Download and prepare the datasets:

- **AV-MNIST**: Image-audio digit classification
- **RAVDESS**: Visual-audio emotion recognition
- **VGGSound-50k**: Visual-audio scene classification
- **CrisisMMD-V2**: Image-text humanitarian classification
- **NYU-Depth-V2**: RGB-depth semantic segmentation

Then, you can generate index meta files for data partitioning by running the `indices_gen.py` file, or download the meta files consistent with ours from [here](https://drive.google.com/drive/folders/11p7GQ9iazVogsImgPvsJjTWNXTCHYCD3?usp=sharing).

We also provide some preprocessed data for [download](https://drive.google.com/drive/folders/11p7GQ9iazVogsImgPvsJjTWNXTCHYCD3?usp=sharing). Alternatively, you can download the original datasets from their respective papers and process them using the code in the `data_preprocess` directory.

### 2. Cross-modal Knowledge Distillation

Run our method:

```python
# Example for RAVDESS dataset (target modality: visual)
python main-MST-Distill.py --database RAVDESS --batch_size 32 --mode m1 --Tmodel 'DSCNN-I' --Smodel 'VisualBranchNet' --AUXmodel 'AudioBranchNet'
```

Run other method:
If you want to run other CMKD methods, you might need to obtain the pre-trained teacher models first.

1. Teacher model training:

   ```python
   # MM Teacher
   python main-T.py --database RAVDESS --batch_size 32 --mode m1 --Tmodel 'DSCNN-I'
   
   # CM Teacher
   python main-S.py --database RAVDESS --batch_size 32 --mode m2 --Smodel 'AudioBranchNet'
   ```

2. Run other CMKD method:

   ```python
   # Example 1: KD (MM->m1)
   python main-KD.py --database RAVDESS --batch_size 32 --mode m1 --Tmodel 'DSCNN-I' --Smodel 'VisualBranchNet' --ckpt_name 'DSCNN-I_weights_file_path'
   
   # Example 2: KD (m2->m1)
   python main-KD-UU.py --database RAVDESS --batch_size 32 --mode m1 --Tmodel 'AudioBranchNet' --Smodel 'VisualBranchNet' --ckpt_name 'AudioBranchNet_weights_file_path'
   ```

### 3. Model Test

Run `test-T.py` or `test-S.py` to test multimodal and unimodal models respectively. The parameter settings follow the same pattern as described above.

---

## ⚙️ Configuration

### Dataset-Specific Parameters

| Dataset      | Batch Size | Learning Rate | Modality (m1-m2) |
| ------------ | :--------: | :-----------: | :--------------: |
| AV-MNIST     |    512     |     1e-4      |   Image-Audio    |
| RAVDESS      |     32     |     1e-4      |   Visual-Audio   |
| VGGSound-50k |    512     |     1e-4      |   Visual-Audio   |
| CrisisMMD-V2 |    512     |     5e-3      |    Image-Text    |
| NYU-Depth-V2 |     6      |     1e-4      |    RGB-Depth     |

You can find the corresponding teacher and student network names in the `get_Tmodules` and `get_Smodules` functions in the `utils.py` file.

### Important Notes

- **Gradient Accumulation**: You can implement gradient accumulation to maintain consistent effective batch sizes when hardware limitations prevent using the recommended batch sizes.
- **Baseline Methods**: Some comparison methods may require different learning rates according to their original papers.
- **Hyperparameter Tuning**: Since our method already achieves good performance with default settings, we did not further optimize the teacher feature layer selection or MaskNet hyperparameters. You can adjust these as needed for your hardware constraints or performance requirements.

---

## 📄 Citation

If you find this work helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{mst-distill2025,
  title={MST-Distill: Mixture of Specialized Teachers for Cross-Modal Knowledge Distillation},
  author={Anonymous Author(s)},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  year={2025},
  publisher={ACM}
}
```

We appreciate your interest in our work and welcome any feedback or contributions to improve this research! 🙏

---

## 📞 Contact

For any questions or issues, please feel free to open an issue in this repository or reach out to us at [gray1y@stu.xidian.edu.cn](mailto:gray1y@stu.xidian.edu.cn).
