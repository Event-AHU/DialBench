# DialBench: Towards Accurate Reading Recognition of Pointer Meter using Large Foundation Models

------------------------------------------------------------------------



## 🔥 Release

- [2026/08/27] **RPM-10K dataset can also be downloaded from** [[**Dropbox**](https://www.dropbox.com/scl/fi/g9xwiur8ew34iovu8qsat/RPM-10K.zip?rlkey=n9d6yltokbt5fofq48lza1zns&st=gycdq03k&dl=0)] 

-   **[2026/04/08] RPM-10K dataset is now publicly available at Baidu Netdisk.**
[https://pan.baidu.com/s/17TGCwMqBx2KHdcZKgrpR0Q?pwd=r2a6]

-   **\[2026/07/21] Model weights is now publicly available.**
[https://pan.baidu.com/s/1abzTBOimkcuqmJZrChhJrg?pwd=64md]
------------------------------------------------------------------------



## ✨ Highlights

-   ✅ **RPM-10K**: large-scale pointer meter reading dataset
-   ✅ **DialBench**: evaluation benchmark for large foundation models
-   ✅ Simple and strong multimodal baseline for pointer meter reading

------------------------------------------------------------------------

## 📦 Dataset: RPM-10K

**RPM-10K** is designed for accurate and robust pointer meter reading.

-   **Scale**: 10,730 images
-   **Focus**: diverse real-world pointer meters

- Download the dataset from [[Dropbox](https://www.dropbox.com/scl/fi/g9xwiur8ew34iovu8qsat/RPM-10K.zip?rlkey=n9d6yltokbt5fofq48lza1zns&st=gycdq03k&dl=0)] 

------------------------------------------------------------------------

## 🧪 Benchmark: DialBench

DialBench provides a comprehensive benchmark for evaluating pointer
meter reading in multimodal LLMs / VLMs.

Features:

-   Multiple metrics: **Acc_ε**, **Acc_θ**, **Ref↓**, **Rel↓**\
-   Comparison to both open-source and closed-source VLMs

------------------------------------------------------------------------

## 🧩 Model Zoo / Weights

-   **Our Model Weights (TBD)**: `()`\

------------------------------------------------------------------------

## 🛠 Installation

### 1. Create conda environment

``` bash
conda create -n dialbench python=3.9
conda activate dialbench
```

### 2. Install from source

``` bash
git clone https://github.com/Event-AHU/DialBench.git
cd DialBench
pip install -e .
```

------------------------------------------------------------------------

## 🚀 Training

Run:

``` bash
bash train.sh
```

- Modify dataset paths in 'caption_builder.py'

  ```python
  datasets['train'] = dataset_cls(
              vis_processor=self.vis_processors["train"],
              text_processor=self.text_processors["train"],
              ann_paths=[os.path.join(storage_path, '')], 
              vis_root=vis_root,
          )
  ```

------------------------------------------------------------------------

## 🧪 Testing / Evaluation

``` bash
bash test.sh
```

-   Evaluation settings are also directly controlled via `test.sh`

------------------------------------------------------------------------

## 📌 Citation

If you find DialBench useful:

``` bibtex
@article{wang2025dialbench,
  title={DialBench: Towards Accurate Reading Recognition of Pointer Meter using Large Foundation Models},
  author={Wang, Futian and Weng, Chaoliu and Wang, Xiao and Chen, Zhen and Zhao, Zhicheng and Tang, Jin},
  journal={arXiv preprint arXiv:2511.21982},
  year={2025}
}
```

------------------------------------------------------------------------

## 🙏 Acknowledgements

-   BLIVA
-   BLIP-2
-   LAVIS
-   All open-source contributors

------------------------------------------------------------------------

## 📄 License

-   Code: **BSD 3-Clause License**
-   Dataset: **TBD**
-   Model weights: **TBD**
