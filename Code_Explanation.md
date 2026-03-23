# Project Code Explanation

This document provides a detailed, line-by-line explanation of the Python code for the "Privacy-Preserving Federated Transformers" project.

---

## `utils.py` - The Project's Utility Belt

### File Purpose
This file contains all the helper functions and classes that are used by the `client.py` and `server.py` scripts. Its main responsibilities are:
1.  **Data Masking:** Anonymizing raw text data to protect user privacy.
2.  **Data Loading:** Preparing the data and splitting it for the different clients.
3.  **Model Operations:** Converting model weights between the PyTorch and Flower formats and saving the final trained model.

### Libraries (Imports)

- `import pandas as pd`: A powerful library for data manipulation. We use it to read and process the CSV file.
- `from sklearn.model_selection import train_test_split`: A function from the scikit-learn library used to easily split data into training and testing sets.
- `from transformers import AutoTokenizer, AutoModelForSequenceClassification`: Components from the Hugging Face `transformers` library. `AutoTokenizer` is used to convert text into numbers for the model, and `AutoModelForSequenceClassification` is used to load the model's architecture.
- `import torch`: The main PyTorch library, providing the core deep learning framework and tensor operations.
- `import os`: A standard Python library for interacting with the operating system, used here for file path operations.
- `import re`: The regular expression library, used for finding and replacing patterns in text (like emails and phone numbers).
- `import spacy`: A major NLP library used here for its powerful Named Entity Recognition (NER) capabilities.

### Code Breakdown

#### `run_masking_pipeline(csv_path)` Function

*   **Summary:** This function is the complete "Phase 1" data processing pipeline. It takes the path to a raw CSV file, cleans the data, and applies NER masking to anonymize sensitive information. It's smart enough to skip the process if the data is already masked.

*   **Line-by-Line:**
    ```python
    def run_masking_pipeline(csv_path):
        # Loads the dataset from the provided CSV file path into a pandas DataFrame.
        df = pd.read_csv(csv_path)

        # Checks if a column named 'text_masked' already exists.
        if "text_masked" in df.columns:
            # If it exists, print a message and return the DataFrame immediately. This prevents re-doing work.
            print("Pre-masked data detected. Skipping masking pipeline.")
            return df

        # If the 'text_masked' column is not found, the full pipeline runs.
        print("Starting Phase 1: Masking Pipeline...")
        try:
            # Tries to load the pre-installed 'en_core_web_sm' SpaCy model.
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If the model isn't found, it prints a message...
            print("Spacy model not found. Downloading...")
            # ...and uses a command to download it automatically.
            spacy.cli.download("en_core_web_sm")
            # After downloading, it loads the model.
            nlp = spacy.load("en_core_web_sm")

        # Removes any duplicate rows from the DataFrame.
        df = df.drop_duplicates()
        # Removes any rows where the 'text' or 'label' columns are empty.
        df = df.dropna(subset=["text", "label"])
        # Cleans the 'text' column: ensures it's a string, removes leading/trailing whitespace,
        # replaces multiple spaces with a single space, and converts to lowercase.
        df["text"] = df["text"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
        
        # Defines a helper function to perform the masking on a single piece of text.
        def mask_ner(text):
            # Processes the text with the SpaCy NLP model to find entities.
            doc = nlp(text)
            masked_text = text
            # Loops through all named entities found by SpaCy (e.g., PERSON, GPE, ORG).
            for ent in doc.ents:
                # Replaces the entity text (e.g., "John Doe") with its label (e.g., "[PERSON]").
                masked_text = re.sub(rf"\b{re.escape(ent.text)}\b", f"[{ent.label_}]", masked_text)
            
            # Uses regular expressions to find and replace email addresses...
            masked_text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", masked_text)
            # ...and phone numbers.
            masked_text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", masked_text)
            
            # Returns the fully masked text.
            return masked_text

        # Applies the 'mask_ner' function to every row in the 'text' column,
        # storing the result in a new 'text_masked' column.
        df["text_masked"] = df["text"].apply(mask_ner)
        
        print("Phase 1: Masking complete.")
        # Returns the fully cleaned and masked DataFrame.
        return df
    ```

---

#### `TextDataset` Class

*   **Summary:** A standard PyTorch `Dataset` class. Its job is to act as a bridge between the raw text data in the DataFrame and the format PyTorch models expect (tensors).

*   **Line-by-Line:**
    ```python
    class TextDataset(torch.utils.data.Dataset):
        # The initializer sets up the dataset's properties.
        def __init__(self, texts, labels, tokenizer_name, max_len=128):
            # Loads the correct tokenizer from Hugging Face based on the model name.
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Stores the list of texts and labels.
            self.texts = texts
            self.labels = labels
            # Stores the maximum length for a tokenized sentence.
            self.max_len = max_len

        # A required method that returns the total number of items in the dataset.
        def __len__(self):
            return len(self.texts)

        # The main worker method. It gets called for each data sample.
        def __getitem__(self, idx):
            # Takes a single piece of text (at index 'idx')...
            # ...and uses the tokenizer to convert it into numerical inputs for the model.
            # It pads or truncates the text to 'max_len'.
            item = self.tokenizer(
                self.texts[idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt" # Returns PyTorch Tensors.
            )
            # The tokenizer returns tensors with an extra dimension, so this line removes it.
            item = {k: v.squeeze(0) for k, v in item.items()}
            # Adds the corresponding label to the item dictionary, a side to it, converting it to a tensor.
            item["labels"] = torch.tensor(int(self.labels[idx]))
            # Returns the final prepared item (e.g., {'input_ids': tensor(...), 'labels': tensor(...)})
            return item
    ```

---

#### `load_and_split_csv(data_path, num_clients=3)` Function

*   **Summary:** This function orchestrates the entire data preparation process for federated learning. It calls the masking pipeline and then splits the resulting clean data among the specified number of clients.

*   **Line-by-Line:**
    ```python
    def load_and_split_csv(data_path, num_clients=3):
        # Calls the masking pipeline we defined earlier. 'df' is now guaranteed to be masked.
        df = run_masking_pipeline(data_path)
        
        # Randomly shuffles all rows in the DataFrame to ensure data is fairly distributed.
        df = df.sample(frac=1).reset_index(drop=True)

        # Creates an empty list to hold the data partitions for each client.
        partitions = []
        # Calculates the size of the data chunk for each client.
        chunk_size = len(df) // num_clients

        # Loops from client ID 0 to the total number of clients.
        for cid in range(num_clients):
            # Slices the DataFrame to get the chunk for the current client ID.
            chunk = df[cid * chunk_size : (cid + 1) * chunk_size]
            # A special case for the last client to make sure it gets all remaining data.
            if cid == num_clients - 1:
                chunk = df[cid * chunk_size : ]

            # Splits the client's chunk into a training set (60%) and a temporary set (40%).
            train, temp = train_test_split(chunk, test_size=0.4, random_state=42)
            # Splits the temporary set into a validation set (20%) and a test set (20%).
            val, test = train_test_split(temp, test_size=0.5, random_state=42)

            # Creates a dictionary holding the lists of texts and labels for the client's train, val, and test sets.
            # Crucially, it uses the 'text_masked' column for the text data.
            partitions.append({
                "train_texts": train["text_masked"].tolist(),
                "train_labels": train["label"].tolist(),
                "val_texts": val["text_masked"].tolist(),
                "val_labels": val["label"].tolist(),
                "test_texts": test["text_masked"].tolist(),
                "test_labels": test["label"].tolist(),
            })

        # Returns the final list of data partitions.
        return partitions
    ```

---

#### Parameter Helper Functions (`get_parameters` and `set_parameters`)

*   **Summary:** These are essential "translator" functions that convert the model's weights (parameters) between the format PyTorch uses and the format Flower requires.

*   **Line-by-Line:**
    ```python
    # Takes a PyTorch model as input.
    def get_parameters(model):
        # Returns a list of the model's parameters, converted to NumPy arrays on the CPU.
        # This is the format Flower expects.
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Takes a PyTorch model and a list of parameters (in Flower's format) as input.
    def set_parameters(model, parameters):
        # Gets the model's current state dictionary (a map of layer names to weights).
        state_dict = model.state_dict()
        new_state_dict = {}
        # Loops through the model's layers and the incoming parameters simultaneously.
        for (key, _), param in zip(state_dict.items(), parameters):
            # Creates a new state dictionary, loading the new parameters as tensors.
            new_state_dict[key] = torch.tensor(param)
        # Loads the newly created state dictionary into the model.
        model.load_state_dict(new_state_dict, strict=True)
    ```

---

#### `save_global_model(...)` Function

*   **Summary:** This function takes the final, trained global model parameters from the server and saves them to the disk as a complete, usable Hugging Face model.

*   **Line-by-Line:**
    ```python
    def save_global_model(model_name, parameters, save_path="global_model"):
        print(f"Saving global model to {save_path}...")
        # Creates a fresh, empty model architecture of the correct type (e.g., BERT, RoBERTa)
        # using the provided 'model_name'. It does not have the trained weights yet.
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Calls the helper function to load the final trained 'parameters' into the empty model.
        set_parameters(model, parameters)
        
        # Uses the standard Hugging Face method to save the complete model
        # (config files, tokenizer info, and the 'model.safetensors' weights file) to the specified path.
        model.save_pretrained(save_path)
        print("Model saved successfully.")
    ```

---
---

## `client.py` - The Local Training Workforce

### File Purpose
This script defines the behavior of a single federated learning client. Each client simulates a user's device (like a mobile phone or personal computer). Its job is to:
1.  Receive the current global model from the server.
2.  Train (fine-tune) that model on its own local, private data.
3.  Evaluate the model's performance on its local test data.
4.  Send the updated model weights and performance metrics back to the server.

### Libraries (Imports)

- `import argparse`: A standard Python library to parse command-line arguments, allowing us to easily specify the client's ID and which model to use.
- `import flwr as fl`: The main Flower library, providing the `NumPyClient` class that our client is built upon.
- `import torch` and `import torch.nn as nn`: The PyTorch library for deep learning. `nn` is the neural network module used for the loss function (`CrossEntropyLoss`).
- `from torch.utils.data import DataLoader`: A PyTorch utility that loads data in batches, which is more memory-efficient for training.
- `from transformers import AutoModelForSequenceClassification`: Used to download the correct transformer model architecture from Hugging Face.
- `from tqdm import tqdm`: A fun utility that creates smart progress bars for loops, showing you the training progress in real-time.
- `from utils import TextDataset, load_and_split_csv`: Imports the dataset class and data loading function we created in `utils.py`.
- `from sklearn.metrics import ...`: Imports specific functions from scikit-learn to calculate performance metrics like F1-score, recall, accuracy, and AUC.
- `import numpy as np`: A fundamental library for numerical operations, used here to help process metrics for calculation.

### Code Breakdown

#### Global `device` variable
*   **Summary:** A standard PyTorch practice to automatically use a GPU if one is available.
*   **Line-by-Line:**
    ```python
    # Checks if PyTorch can access a CUDA-enabled GPU.
    # If yes, sets 'device' to "cuda". If not, sets it to "cpu".
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```

---

#### `FlowerClient` Class
*   **Summary:** This class defines all the logic and behavior for a federated learning client. It tells Flower how to get model parameters, how to train the model (`fit`), and how to evaluate it (`evaluate`).

*   **`__init__(...)` Method (The Initializer)**
    *   **Summary:** Sets up the client's initial state, storing its ID, model, data, and creating the necessary components for training.
    *   **Line-by-Line:**
        ```python
        def __init__(self, cid, model, tokenizer_name, client_data, batch_size=8, local_epochs=1, lr=2e-5):
            # Stores the unique client ID (e.g., 0, 1, or 2).
            self.cid = cid
            # Stores the model object and moves it to the active device (GPU or CPU).
            self.model = model.to(device)
            # Stores the name of the tokenizer (which is the same as the model name).
            self.tokenizer_name = tokenizer_name
            # Stores the dictionary containing this client's specific data partition.
            self.client_data = client_data
            # Stores training hyperparameters: batch size, number of local epochs, and learning rate.
            self.batch_size = batch_size
            self.local_epochs = local_epochs
            self.lr = lr
            # Creates an instance of the CrossEntropyLoss function, the "critic" that measures model error.
            self.criterion = nn.CrossEntropyLoss()

            # Creates the PyTorch Dataset objects for this client's training and testing data.
            self.train_ds = TextDataset(client_data["train_texts"], client_data["train_labels"], tokenizer_name)
            self.test_ds = TextDataset(client_data["test_texts"], client_data["test_labels"], tokenizer_name)
        ```

*   **`get_parameters(...)` and `set_parameters(...)` Methods**
    *   **Summary:** These two methods are the client's main way of communicating with the server. `set_parameters` receives the new global model from the server, and `get_parameters` sends the locally trained model back.
    *   **Line-by-Line:**
        ```python
        # This method is called by the server to request the client's model weights.
        def get_parameters(self, config):
            # It returns a list of the model's parameters, converted to NumPy arrays.
            return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

        # This method is called by the server to give the client the new global model weights.
        def set_parameters(self, parameters):
            # It gets the model's current state dictionary (a map of layer names to weights).
            state_dict = self.model.state_dict()
            new_state_dict = {}
            # It loops through the model's layers and the incoming parameters simultaneously...
            for (key, _), param in zip(state_dict.keys(), parameters):
                # ...creating a new state dictionary with the updated weights.
                new_state_dict[key] = torch.tensor(param).type_as(state_dict[k])
            # It loads the new weights into the local model.
            self.model.load_state_dict(new_state_dict, strict=True)
        ```

*   **`fit(...)` Method (Local Training)**
    *   **Summary:** This is the core training method. The server calls this function to tell the client: "Train the model I'm giving you on your local data."
    *   **Line-by-Line:**
        ```python
        def fit(self, parameters, config):
            # First, update the local model with the global weights from the server.
            self.set_parameters(parameters)
            # Set the model to training mode.
            self.model.train()
            # Create the AdamW optimizer, the "coach" that updates the model's weights.
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

            # Create a DataLoader to efficiently feed data to the model in batches.
            train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

            # Loop through the entire training dataset for the specified number of 'local_epochs'.
            for epoch in range(self.local_epochs):
                # 'tqdm' creates a nice progress bar for the training loop.
                loop = tqdm(train_loader, desc=f"Client {self.cid} Epoch {epoch+1}", leave=False)
                # For each batch of data in the training set...
                for batch in loop:
                    # Clear any old error calculations.
                    optimizer.zero_grad()
                    # Prepare the inputs and move them to the GPU/CPU.
                    inputs = {k:v.to(device) for k,v in batch.items() if k != "labels"}
                    labels = batch["labels"].to(device)
                    # **Forward Pass**: Get the model's raw output (logits).
                    outputs = self.model(**inputs)
                    # Calculate the error (loss) between the model's prediction and the true labels.
                    loss = self.criterion(outputs.logits, labels)
                    # **Backward Pass**: Calculate how much each weight contributed to the error.
                    loss.backward()
                    # **Optimizer Step**: Update the weights to reduce the error.
                    optimizer.step()

            # After training, return three items to the server:
            # 1. The newly trained local model weights.
            # 2. The number of samples this client trained on.
            # 3. A dictionary of performance metrics from a fresh evaluation on the test set.
            return self.get_parameters(config), len(self.train_ds), self.evaluate_local()
        ```

*   **`evaluate(...)` Method (Called by Server)**
    *   **Summary:** A simple method called by the server to trigger a local evaluation.
    *   **Line-by-Line:**
        ```python
        def evaluate(self, parameters, config):
            # Update the model with the weights from the server.
            self.set_parameters(parameters)
            # Return the results of a local evaluation. The format is (loss, num_examples, metrics).
            # We return 0.0 for loss as the main metrics are in the dictionary.
            return 0.0, len(self.test_ds), self.evaluate_local()
        ```

*   **`evaluate_local()` Method (The Metric Calculator)**
    *   **Summary:** This is the powerful new method that performs a detailed evaluation of the model on the client's local test set.
    *   **Line-by-Line:**
        ```python
        def evaluate_local(self):
            # Set the model to evaluation mode (disables things like dropout).
            self.model.eval()
            # Create a DataLoader for the test set.
            loader = DataLoader(self.test_ds, batch_size=self.batch_size)
            # Create empty lists to store all labels and predictions.
            all_labels, all_preds, all_probs = [], [], []

            # Disable gradient calculations to save memory and speed up evaluation.
            with torch.no_grad():
                # Loop through every batch in the test set.
                for batch in loader:
                    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                    labels = batch["labels"].to(device, dtype=torch.long)
                    # Get the model's raw output.
                    outputs = self.model(**inputs)
                    # Find the predicted class by taking the index of the highest logit.
                    preds = torch.argmax(outputs.logits, dim=1)
                    # Apply softmax to get prediction probabilities (for AUC).
                    probs = torch.softmax(outputs.logits, dim=1)

                    # Add the batch results to our master lists.
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            # Use scikit-learn to calculate all the required metrics.
            acc = float(accuracy_score(all_labels, all_preds))
            # 'average="macro"' is important for calculating the metric independently for each class and then taking the average.
            f1 = float(f1_score(all_labels, all_preds, average="macro"))
            recall = float(recall_score(all_labels, all_preds, average="macro"))

            # The 'try/except' block handles the case where AUC can't be calculated (e.g., only one class in a batch).
            try:
                # Logic to calculate AUC differently for binary vs. multi-class cases.
                if len(np.unique(all_labels)) > 2:
                    auc = float(roc_auc_score(np.eye(len(np.unique(all_labels)))[all_labels], all_probs, average="macro", multi_class="ovr"))
                else:
                    auc = float(roc_auc_score(all_labels, np.array(all_probs)[:, 1]))
            except ValueError:
                auc = 0.0 # If AUC calculation fails, default to 0.0.

            # Create the final metrics dictionary.
            metrics = {"acc": acc, "f1": f1, "recall": recall, "auc": auc}
            # Print the results to the client's console.
            print(f"
[Client {self.cid}] Metrics: Acc={acc:.4f}, F1={f1:.4f}, Recall={recall:.4f}, AUC={auc:.4f}")
            # Return the dictionary.
            return metrics
        ```

---

#### `if __name__ == "__main__":` Block (The Script's Entry Point)
*   **Summary:** This code runs only when you execute `python client.py` from the command line. It sets up and starts the client.
*   **Line-by-Line:**
    ```python
    if __name__ == "__main__":
        # Sets up the command-line argument parser.
        parser = argparse.ArgumentParser()
        # Defines the arguments the script accepts: client ID, number of clients, model name, and data path.
        parser.add_argument("--cid", type=int, required=True)
        parser.add_argument("--num_clients", type=int, default=3)
        parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
        parser.add_argument("--data_path", type=str, default="masked_output_ner_test.csv")
        args = parser.parse_args()

        # Calls the main data loading function from utils.py. This triggers the masking and splitting.
        partitions = load_and_split_csv(args.data_path, args.num_clients)
        # Selects the data partition for this specific client based on its --cid.
        client_data = partitions[args.cid]

        # Determines the number of unique labels from the data (e.g., 2 for 'Related'/'Not Related').
        num_labels = len(set(client_data["train_labels"]))

        # **Downloads the pre-trained model architecture from Hugging Face** based on the --model_name argument.
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels
        )

        # **Starts the Flower client**, connecting to the server and beginning the FL process.
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=FlowerClient(args.cid, model, args.model_name, client_data)
        )
    ```

---
---

## `server.py` - The Central Coordinator

### File Purpose
This script is the "brain" of the federated learning operation. It orchestrates the entire process:
1.  Starts a server and waits for clients to connect.
2.  Tells the clients when to start training and evaluating.
3.  Gathers the updated models and performance metrics from the clients after each round.
4.  Aggregates the model updates to create an improved global model.
5.  Aggregates the metrics to provide a system-wide view of performance.
6.  Saves the final model and performance reports.

### Libraries (Imports)
- `import flwr as fl`: The main Flower library used to start the server and define the strategy.
- `from typing import ...`: Standard Python types used for type hinting, making the code more readable.
- `from utils import save_global_model`: Imports the helper function for saving the final model from `utils.py`.
- `import numpy as np`: Used by Flower for handling model parameters.
- `import argparse`: To read command-line arguments, specifically the `--model_name`.
- `import json`: To save the performance metrics in a structured JSON file.
- `import os`: To create the results directory.

### Code Breakdown

#### Global Setup
*   **Summary:** Creates a directory to store all the results from the experiments.
*   **Line-by-Line:**
    ```python
    # Defines the name for the folder that will store all results.
    RESULTS_DIR = "FL_Project_Results"
    # Creates this directory if it doesn't already exist.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ```

---

#### `aggregate_eval_metrics(...)` Function
*   **Summary:** This is a crucial new function that tells the server how to average the detailed metric dictionaries received from the clients.
*   **Line-by-Line:**
    ```python
    def aggregate_eval_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        # Calculates the total number of examples evaluated across all clients.
        total_examples = sum(n for n, _ in metrics)
        # If no examples were evaluated, return an empty dictionary to avoid errors.
        if total_examples == 0:
            return {}

        # Defines a helper function to calculate the weighted average for a given metric key (e.g., "acc").
        def wavg(key: str) -> float:
            # It multiplies each client's metric value by its number of examples, sums them up,
            # and divides by the total number of examples.
            return sum(m[key] * n for n, m in metrics) / total_examples

        # Returns a new dictionary containing the aggregated, system-wide score for each metric.
        return {
            "acc": wavg("acc"),
            "f1": wavg("f1"),
            "recall": wavg("recall"),
            "auc": wavg("auc"),
        }
    ```

---

#### `save_metrics(...)` Function
*   **Summary:** A simple helper function to save the aggregated metrics to a JSON file.
*   **Line-by-Line:**
    ```python
    def save_metrics(metrics: Dict[str, Scalar], round_num: int, model_name: str):
        # Creates a unique file path for the metrics file based on the model and round number.
        path = os.path.join(RESULTS_DIR, f"metrics_{model_name.replace('/', '_')}_round_{round_num}.json")
        # Opens the file in write mode.
        with open(path, "w") as f:
            # Uses the json library to dump the metrics dictionary into the file with nice formatting.
            json.dump(metrics, f, indent=4)
        print(f"
[Server] Metrics saved to {path}")
    ```

---

#### `SaveModelStrategy` Class
*   **Summary:** This is your custom server strategy. It inherits from Flower's standard `FedAvg` but adds your specific logic for saving the model and metrics.
*   **`__init__(...)` Method**
    *   **Summary:** Initializes the strategy, storing the model name for later use.
    *   **Line-by-Line:**
        ```python
        def __init__(self, model_name, *args, **kwargs):
            # Calls the parent class's initializer.
            super().__init__(*args, **kwargs)
            # Stores the model name passed in from the command line.
            self.model_name = model_name
        ```
*   **`aggregate_fit(...)` Method**
    *   **Summary:** This method is called by Flower after each training round to aggregate results. You've customized it to save metrics and the final model.
    *   **Line-by-Line:**
        ```python
        def aggregate_fit(self, server_round, results, failures):
            # First, call the default FedAvg aggregation logic to get the new global model and aggregated metrics.
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            # If the clients returned any metrics (your new clients do)...
            if aggregated_metrics:
                # ...save them to a JSON file.
                save_metrics(aggregated_metrics, server_round, self.model_name)

            # Check if aggregation was successful and if it's the final round (round 3).
            if aggregated_parameters is not None and server_round == 3:
                print("
[Server] Final round completed. Saving global model...")
                # Convert the parameters to the right format.
                weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
                # Create the unique save path inside the results directory.
                save_path = os.path.join(RESULTS_DIR, f"final_{self.model_name.replace('/', '_')}_model")
                # Call the helper function to save the final model to disk.
                save_global_model(self.model_name, weights, save_path)

            # Return the aggregated parameters and metrics for the next round.
            return aggregated_parameters, aggregated_metrics
        ```

---

#### `if __name__ == "__main__":` Block (The Script's Entry Point)
*   **Summary:** This code runs when you execute `python server.py`. It configures and starts the entire federated learning server.
*   **Line-by-Line:**
    ```python
    if __name__ == "__main__":
        # Sets up the command-line argument parser.
        parser = argparse.ArgumentParser()
        # Defines the --model_name argument.
        parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Name of the model to train")
        args = parser.parse_args()

        # **This is the main configuration block.**
        # It creates an instance of your custom SaveModelStrategy.
        strategy = SaveModelStrategy(
            # Passes the model name to the strategy.
            model_name=args.model_name,
            # These arguments tell the server to use 100% of available clients for training and evaluation.
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            # These arguments tell the server to wait for at least 3 clients to be available,
            # and to use at least 3 for training and 3 for evaluation.
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            # **This is the key line that "activates" your advanced metric aggregation.**
            # It tells Flower to use your custom function to average the metrics from evaluation rounds.
            evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
        )

        # **Starts the Flower server** with the specified address, number of rounds, and your fully configured strategy.
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=strategy,
        )
    ```
