# 🇺🇦 UA News Cities Search

This project focuses on analyzing Ukrainian news datasets, extracting entities (specifically cities), and fine-tuning Named Entity Recognition (NER) models using HuggingFace and SpaCy.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| **`data_analysis.ipynb`** | Pre-processing phase for name/articles analysis and GPU-accelerated n-gram word search. |
| **`train.ipynb`** | Fine-tuning of the HuggingFace model using `data/test_dataset.json` and storing the results. |
| **`spacy_inference.ipynb`** | Loading the fine-tuned model's weights into the spaCy infrastructure for inference. |

## Metrics & Performance

<img src="https://github.com/user-attachments/assets/0debd62d-1fd1-46ee-ba21-6b20ed0f0052" alt="metrics_dashboard_aggressive" width="100%" />

* [**Google Colab Version**](https://drive.google.com/drive/folders/1_Gto0bpCgU3BaYSuaDUt7ZEHKASP3Fb_?usp=drive_link) – Access the notebooks and drive structure.

### Datasets
The project utilizes the following datasets for training and analysis:
* [**Ukrainian Fake and True News** (Kaggle)](https://www.kaggle.com/datasets/zepopo/ukrainian-fake-and-true-news?resource=download)
* [**Ukrainian News** (HuggingFace)](https://huggingface.co/datasets/zeusfsx/ukrainian-news)

---
<div align="center">
    ☮️
</div>
