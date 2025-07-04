# 🧾 TextTabBench: A Dataset Pool for Text-Tabular Learning

This repository provides a curated pool of datasets designed for evaluating models that combine structured **tabular features** and unstructured **text fields**. These datasets support supervised learning tasks including regression and binary/multiclass classifcation.

This work is part of our workshop paper:

📄 **[Workshop Paper on OpenReview](https://openreview.net/pdf?id=yrmoQG9NAV)**

---

## 📁 Repository Structure
TEXTTABBENCH/
│
├── configs/ # Configuration scripts for each dataset
├── datasets_files/ # Raw or preprocessed dataset files
├── datasets_notebooks/ # Jupyter notebooks to load and inspect datasets
│
├── paper_datasets/ # Datasets included in our workshop paper
│ ├── classification/ # Datasets with categorical targets
│ └── regression/ # Datasets with continuous targets
│
├── extra_datasets/ # Additional datasets not included in the paper
│
├── examples/ # Example scripts using the datasets
│
├── src/ # Source code for preprocessing and utilities
│ ├── dataloader_functions/
│ └── text_processors/
│
├── requirements.txt # Python dependencies
└── README.md # You are here
---

## 📚 Dataset Groups

### `paper_datasets/` (Main Benchmark)

These datasets were selected and evaluated in our workshop paper. They were chosen based on:

- Balanced and meaningful text features
- Real-world tabular context
- Predictive signal beyond text-only baselines

They are split into `classification/` and `regression/` folders. Each dataset is provided as a Jupyter notebook with all required steps for inspection and use.

### `extra_datasets/` (Valuable Additions)

These are high-quality datasets that did not make it into the final benchmark due to reasons like domain overlap, preprocessing inconsistencies, or weaker text signals. Still, they are useful for experimentation and ablation studies.

### Future Additions

Some datasets that narrowly missed the benchmark threshold may be added later under custom folders such as:

- `near_threshold_datasets/`
- `text_too_short/`
- `exploratory_datasets/`

These are still useful under specific research contexts.

---

## 🚀 Getting Started

To explore and use the datasets in this repository, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/TextTabBench.git
cd TextTabBench
```
### 2. Set Up the Environment
```bash
pip install -r requirements.txt
```
### 3. Explore the Datasets
Open any dataset notebook to go throught the download and necessary preprocessing. You can also view the data at different stages of processing.
At last, there is also space for some further analysis of the final data, ready to be ussed by any table-w-text solution pipeline.

### 4. Download the Datasets
Run `src/download_datasets/download_datasets.py --dataset <dataset_name> --path <home_dir_for_downloads> --task <reg/clf> --selection <all/default>` to download either a single dataset or a whole subset of them.

---

## 🤝 Contributing

We welcome contributions to this dataset pool!

If you'd like to add a new dataset or improve existing ones, please follow these guidelines:

- ✅ Provide a short description of the dataset
- 🏷️ Clearly identify:
  - The **target** column
  - The **text** feature(s)
- 📓 Include a Jupyter notebook following the structure of existing examples
- 💬 Shortly explain why the dataset is valuable for evaluating text-tabular models

To contribute:
1. Fork the repository
2. Create a new branch
3. Add your dataset and notebook
4. Submit a pull request

---

## 📬 Citation

If you use this dataset pool in your research or build on our benchmark, please cite the following workshop paper:

> **Towards Benchmarking Foundation Models for Tabular Data With Text**  
> OpenReview, 2025
> [https://openreview.net/pdf?id=yrmoQG9NAV](https://openreview.net/pdf?id=yrmoQG9NAV)

BibTeX:
```bibtex
@inproceedings{TextTabBench2024,
  title={Towards Benchmarking Foundation Models for Tabular Data With Text},
  author={Mraz, Das, Gupta and others},
  booktitle={ICML 2025 Workshop on Foundation Models for Structured Data (FMSD)},
  year={2025},
  url={https://openreview.net/pdf?id=yrmoQG9NAV}
}
```