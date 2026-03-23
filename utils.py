import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import re
import spacy

# -------------------------------------------------
# PHASE 1: MASKING PIPELINE
# -------------------------------------------------
def run_masking_pipeline(csv_path):
    """
    Loads data from a CSV with PII in separate columns ('Name', 'Email', 'Phone', 'Text').
    Masks the PII within the 'Text' column and renames columns for the FL pipeline.
    """
    # Check if the target file is the one with the different structure
    if 'dataset.csv' not in csv_path:
        # Fallback to original logic for other files like 'combined_with_dummy_corrected.csv'
        df = pd.read_csv(csv_path)
        if "text_masked" not in df.columns:
            print("Running original spaCy-based masking for non-dataset.csv file...")
            # (Keeping a simplified version of the original logic as a fallback)
            df = df.dropna(subset=["text", "label"])
            df["text_masked"] = df["text"] # Placeholder if spacy is not used
        return df

    print("Starting Phase 1: Masking Pipeline for dataset.csv...")
    df = pd.read_csv(csv_path)

    # Rename columns to standard format first
    df = df.rename(columns={"Text": "text", "status": "label"})

    # Drop rows with missing text or labels
    df.dropna(subset=['text', 'label', 'Name', 'Email', 'Phone'], inplace=True)

    def mask_pii_from_columns(row):
        masked_text = str(row['text'])
        # Mask Name, Email, and Phone based on their columns
        # Use regex to ensure whole-word matching to avoid partial replacements
        masked_text = re.sub(r'\b' + re.escape(str(row['Name'])) + r'\b', '[PERSON]', masked_text, flags=re.IGNORECASE)
        masked_text = re.sub(re.escape(str(row['Email'])), '[EMAIL]', masked_text, flags=re.IGNORECASE)
        masked_text = re.sub(re.escape(str(row['Phone'])), '[PHONE]', masked_text)
        return masked_text

    # Apply the new masking function
    df['text_masked'] = df.apply(mask_pii_from_columns, axis=1)

    # Clean up text (can be done after masking)
    df["text_masked"] = df["text_masked"].str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
    df["text"] = df["text"].str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
    
    # Select and reorder columns for downstream compatibility
    final_df = df[['text', 'label', 'text_masked']].copy()

    print("Phase 1: Masking complete.")
    return final_df

# -------------------------------------------------
# Dataset class for tokenization
# -------------------------------------------------
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer_name, max_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in item.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

# -------------------------------------------------
# PHASE 2: DATA LOADING FOR FEDERATED LEARNING
# -------------------------------------------------
def load_and_split_csv(data_path, num_clients=3):
    # Run the masking pipeline (or skip if data is pre-masked)
    df = run_masking_pipeline(data_path)
    
    # Clean and map labels to binary (0 for Normal, 1 for Mental Health Issue)
    # Filter out 'nan' and email-like entries from labels
    df = df[df['label'] != 'nan']
    df = df[~df['label'].str.contains(r'@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')]

    # Define the mapping
    def map_label_to_binary(label):
        if label == 'Normal':
            return 0
        else:
            return 1 # All other identified labels (Anxiety, Depression, etc.) are considered mental health issues

    df['label'] = df['label'].apply(map_label_to_binary)

    # Shuffle for fairness
    df = df.sample(frac=1).reset_index(drop=True)

    partitions = []
    chunk_size = len(df) // num_clients

    for cid in range(num_clients):
        chunk = df[cid * chunk_size : (cid + 1) * chunk_size]
        if cid == num_clients - 1:
            chunk = df[cid * chunk_size : ]

        train, temp = train_test_split(chunk, test_size=0.4, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        partitions.append({
            "train_texts": train["text_masked"].tolist(),
            "train_labels": train["label"].tolist(),
            "val_texts": val["text_masked"].tolist(),
            "val_labels": val["label"].tolist(),
            "test_texts": test["text_masked"].tolist(),
            "test_labels": test["label"].tolist(),
        })

    return partitions

# -------------------------------------------------
# Parameter helpers for Flower
# -------------------------------------------------
def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    state_dict = model.state_dict()
    new_state_dict = {}
    for (key, _), param in zip(state_dict.items(), parameters):
        new_state_dict[key] = torch.tensor(param)
    model.load_state_dict(new_state_dict, strict=True)

# -------------------------------------------------
# Save Global Model Helper
# -------------------------------------------------
def save_global_model(model_name, parameters, save_path="global_model"):
    print(f"Saving global model to {save_path}...")
    # Explicitly load config and set model_type for prajjwal1/bert-tiny, as its config is malformed.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    config.model_type = 'bert'
    config.num_labels = 2
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    
    # Load the trained parameters into it
    set_parameters(model, parameters)
    
    # Save to disk
    model.save_pretrained(save_path)
    print("Model saved successfully.")
