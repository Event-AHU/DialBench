# DialBench: Towards Accurate Reading Recognition of Pointer Meter using Large Foundation Models

------------------------------------------------------------------------

## 🔥 Release

-   **\[2025/xx/xx\]** DialBench project initialized and code released\
-   **\[2025/xx/xx\]** **RPM-10K dataset** & **DialBench benchmark**
    released\
-   **\[TBD\]** Model weights coming soon

------------------------------------------------------------------------

## ✨ Highlights

-   ✅ **RPM-10K**: large-scale pointer meter reading dataset\
-   ✅ **DialBench**: evaluation benchmark for large foundation models\
-   ✅ Simple and strong multimodal baseline for pointer meter reading\

------------------------------------------------------------------------

## 📦 Dataset: RPM-10K

**RPM-10K** is designed for accurate and robust pointer meter reading.

-   **Scale**: 10,000 images\
-   **Focus**: diverse real-world pointer meters\

> 你可以在这里继续补充：数据结构、下载方式、标注格式、示例图等。

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
    \> 发布后将括号替换为真实链接（HuggingFace / Google Drive /
    ModelScope 等）

------------------------------------------------------------------------

## 📊 Experimental Results

### Pointer Meter Reading on \*\*`\datasetname*`{=tex}\*

> † denotes closed-source visual language models.

(LaTeX table unchanged---GitHub will render as code block.)

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
@misc{your2025dialbench,
  title={DialBench: Towards Accurate Reading Recognition of Pointer Meter using Large Foundation Models},
  author={Your Name and ...},
  year={2025},
  eprint={xxxx.xxxxx},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

------------------------------------------------------------------------

## 🙏 Acknowledgements

-   BLIVA\
-   BLIP-2\
-   LAVIS\
-   All open-source contributors

------------------------------------------------------------------------

## 📄 License

-   Code: **BSD 3-Clause License**\
-   Dataset: **TBD**\
-   Model weights: **TBD**
