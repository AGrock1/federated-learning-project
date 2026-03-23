# Combined Federated Learning Pipeline

This project contains a complete, unified pipeline that performs two main phases:
1.  **Phase 1 (Masking):** It first takes a raw dataset and applies a data cleaning and NER masking process using `spaCy`. If the data is already masked, it skips this step.
2.  **Phase 2 (Federated Learning):** It then uses the masked data to train a model using Federated Learning with Flower and Hugging Face Transformers.

This pipeline allows you to go from raw or pre-masked data to a trained federated model in a single, streamlined process.

## How to Run

### 1. Navigate to the Directory

Open your terminal and change to this project's directory:

```bash
cd C:\Users\ghazaesearch\Combined_Federated_Pipeline
```

### 2. Set Up a Virtual Environment

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages.

```bash
pip install flwr torch transformers pandas scikit-learn spacy
```

### 4. Download the SpaCy Model

If you are using a raw, unmasked dataset, the script will require a `spaCy` model. You can pre-download it to speed up the first run:

```bash
python -m spacy download en_core_web_sm
```

### 5. Start the Federated Learning Process

You will need to open several terminals.

**A. In the first terminal, start the server:**

You can optionally specify which base model to use for the global model.

```bash
# Example with the default model
python server.py

# Or, specify a different model
python server.py --model_name bert-base-uncased
```

**B. In separate terminals, start the clients:**

Point the `--data_path` argument to your data file (the script will detect if it's raw or pre-masked). Remember to also specify the `--model_name`.

*   **Client 0:**
    ```bash
    python client.py --cid 0 --model_name distilbert-base-uncased --data_path path/to/your/data.csv
    ```
*   **Client 1:**
    ```bash
    python client.py --cid 1 --model_name distilbert-base-uncased --data_path path/to/your/data.csv
    ```
*   **Client 2:**
    ```bash
    python client.py --cid 2 --model_name distilbert-base-uncased --data_path path/to/your/data.csv
    ```

Replace `path/to/your/data.csv` with the actual path to your dataset. The program will handle the rest.
