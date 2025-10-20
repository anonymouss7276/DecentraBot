# DecentraBot: Reproducibility Package for TCSS

This repository contains the code and data necessary to reproduce the core architecture of the DecentraBot system, as described in our paper submitted to IEEE TCSS. The primary artifact is the `DecentraBot_Pipeline.ipynb` notebook, which provides a complete, end-to-end implementation of the system using sample data and an open-source LLM.

## Overview 

The notebook demonstrates the full architectural pipeline, including:
- Data loading and sampling from the `NFT-70M` and `DecentraFAQ` datasets.
- Training of the **Query Classifier** (DistilBERT) for intent detection.
- Data preparation and fine-tuning of the **Multitask-TabLLM** (Mistral-7B adapter) for regression, classification, and forecasting tasks on tabular data.
- Implementation of the **RAG pipeline** for contextual augmentation using a synthesized knowledge base.
- A full demonstration of the sophisticated **What-Next Prompting (WNP)** logic using a local, base Mistral-7B model for response generation.

## Setup Instructions 

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Samy-53-ab/DecentraBot-TCSS-Reproducibility.git](https://github.com/Samy-53-ab/DecentraBot-TCSS-Reproducibility.git)
    cd DecentraBot-TCSS-Reproducibility
    ```

2.  **Create a Python Environment:**
    Using a virtual environment (like `venv` or `conda`) is highly recommended. Requires Python 3.10+.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    This installs the specific library versions known to be compatible with this notebook.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run 

The entire pipeline can be executed by running the `DecentraBot_Pipeline.ipynb` notebook cell by cell.

-   **Recommended Environment:** Google Colab with a GPU runtime (T4 or higher is strongly recommended) or a local machine with a suitable NVIDIA GPU (>= 16GB VRAM recommended) and sufficient RAM (>16GB). Running on CPU is possible but will be extremely slow.
-   The notebook is self-contained and uses the provided sample data and trained model artifacts located in the `/artifacts` directory. Ensure this directory is present.
-   Fine-tuning steps include checkpointing and will skip retraining if a previously completed run is detected.

## Scope of this Reproducibility Package 

This repository provides a verifiable and executable implementation for review purposes. To ensure accessibility and efficient execution, the following choices were made:

* **Generative Models:** We employ the `mistralai/Mistral-7B-Instruct-v0.2` open-source model (loaded with 4-bit quantization via `bitsandbytes`) for both the core conversational logic (WNP execution, final response) and as the base for the fine-tuned Multitask-TabLLM. This ensures the pipeline runs locally without external API dependencies, guaranteeing reproducibility.
* **Dataset Scope:** The pipeline operates on **representative samples** (20k NFT transactions, 1k FAQ entries) included in `/artifacts`. This allows for rapid execution of all training and inference steps. The purpose is to validate the architectural flow and component functionality, not to replicate the exact performance metrics from the full-scale experiments in the paper.
* **Lightweight Knowledge Base:** The RAG pipeline uses a small, synthesized knowledge base to demonstrate the retrieval mechanism effectively without requiring a large vector database setup.

## Provided Artifacts & Hyperparameters 

All necessary data samples and pre-trained model adapters are included in the `/artifacts` directory:

-   `/artifacts/nft_sample_data.csv`: 20k-row sample from `NFT-70M`.
-   `/artifacts/faq_sample_data.csv`: 1k-row sample from `DecentraFAQ`.
-   `/artifacts/query_classifier_model/`: Fine-tuned DistilBERT model for intent classification.
-   `/artifacts/multitask_tabllm_adapter/`: Fine-tuned PEFT/LoRA adapter weights for the Multitask-TabLLM (Mistral-7B base).

#### Key Hyperparameters
-   **Query Classifier (DistilBERT):**
    -   Model: `distilbert-base-uncased`, Epochs: 3, Batch Size: 16
-   **Multitask-TabLLM (Mistral-7B Fine-tuning):**
    -   Base Model: `mistralai/Mistral-7B-Instruct-v0.2`
    -   Quantization: 4-bit (`bitsandbytes`)
    -   PEFT Method: LoRA (`r=64`, `lora_alpha=16`)
    -   Max Steps: 100, Learning Rate: 2e-4, Batch Size (effective): 8
-   **RAG:**
    -   Embedding Model: `all-MiniLM-L6-v2`
    -   Index: FAISS `IndexFlatL2`
