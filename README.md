# DecentraBot: Reproducibility Package for TCSS

This repository contains the code and data necessary to reproduce the core architecture of the DecentraBot system, as described in our paper submitted to IEEE TCSS. The primary artifact is the `DecentraBot_Pipeline.ipynb` notebook, which provides a complete, end-to-end implementation of the system.

## Overview

The notebook demonstrates the full architectural pipeline, including:
- Data loading and preparation using sample datasets.
- Training of the **Query Classifier** (DistilBERT) for intent detection.
- Data preparation and fine-tuning of the **Multitask-TabLLM** (a Mistral-7B adapter) for tabular prediction tasks.
- Implementation of the **RAG pipeline** for contextual augmentation.
- A full demonstration of the sophisticated **What-Next Prompting (WNP)** logic using a local, open-source LLM.

## How to Run in Google Colab (Recommended)

This project is optimized for Google Colab, which provides free access to the necessary GPU hardware.

### 1. Set Up Your Google Drive

The notebook requires access to one data file from Google Drive to initialize the process.
- In your Google Drive, create a root folder named `DecentraBot_Project`.
- **Upload the `DecentraFAQ.csv` file** (provided in this repository) into the `DecentraBot_Project` folder.

Your Google Drive structure should now look like this:
My Drive/ └── DecentraBot_Project/ └── decentraFAQ.csv

The notebook will automatically create an `artifacts` subfolder within this directory to save all trained models and data samples during execution.

### 2. Run the Notebook in Colab

1.  **Upload the Notebook:** Open [Google Colab](https://colab.research.google.com/), and select **File → Upload notebook...**. Choose the `DecentraBot_Pipeline.ipynb` file from this repository.
2.  **Select GPU Runtime:** In the notebook menu, navigate to **Runtime → Change runtime type**. From the "Hardware accelerator" dropdown, select **T4 GPU** (or a higher tier like A100 if available). This step is crucial for performance.
3.  **Run All Cells:** In the menu, select **Runtime → Run all**. The notebook will execute from top to bottom. It will first install all necessary libraries, then mount your Google Drive, and proceed to train all components and run the final demonstration. The entire process should complete without requiring further user intervention.

## Local Setup Instructions (Alternative)

For users who wish to run the code on a local machine with appropriate hardware:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/anonymouss7276/DecentraBot.git](https://github.com/anonymouss7276/DecentraBot.git)
    cd DecentraBot
    ```

2.  **Hardware Requirements:** A local machine with a suitable NVIDIA GPU (>= 16GB VRAM) and system RAM (>16GB) is strongly recommended.

3.  **Create a Python Environment:**
    Requires Python 3.10+. Using a virtual environment is highly recommended.
    ```bash
    python -m venv venv
    source ven v/bin/activate  # On Linux/macOS
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Launch Jupyter:**
    ```bash
    jupyter notebook DecentraBot_Pipeline.ipynb
    ```

## Scope of this Reproducibility Package

This repository provides a verifiable implementation for review purposes. To ensure accessibility and efficient execution, the following choices were made:

* **Generative Models:** We employ the `mistralai/Mistral-7B-Instruct-v0.2` open-source model (loaded with 4-bit quantization) for all generative tasks. This ensures the pipeline runs without external API dependencies, guaranteeing reproducibility.
* **Dataset Scope:** The pipeline operates on representative samples of the original datasets. This allows for rapid execution of all training and inference steps. The purpose is to validate the architectural flow, not to replicate the exact performance metrics from the full-scale experiments in the paper.
* **Lightweight Knowledge Base:** The RAG pipeline uses a small, synthesized knowledge base to demonstrate the retrieval mechanism effectively.

## Generated Artifacts & Hyperparameters

Running the notebook will generate all necessary artifacts in the `/artifacts` subfolder of your Google Drive project directory.

#### Key Hyperparameters
-   **Query Classifier (DistilBERT):**
    -   Model: `distilbert-base-uncased`, Epochs: 3, Batch Size: 16
-   **Multitask-TabLLM (Mistral-7B Fine-tuning):**
    -   Base Model: `mistralai/Mistral-7B-Instruct-v0.2`
    -   Quantization: 4-bit (`bitsandbytes`)
    -   PEFT Method: LoRA (`r=64`, `lora_alpha=16`)
    -   Max Steps: 100, Learning Rate: 2e-4
-   **RAG:**

    -   Embedding Model: `all-MiniLM-L6-v2`
