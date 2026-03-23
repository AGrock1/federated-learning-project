import argparse
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from utils import TextDataset, load_and_split_csv
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, tokenizer_name, client_data, batch_size=8, local_epochs=1, lr=2e-5):
        self.cid = cid
        self.model = model.to(device)
        self.tokenizer_name = tokenizer_name
        self.client_data = client_data
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        # Datasets
        self.train_ds = TextDataset(client_data["train_texts"], client_data["train_labels"], tokenizer_name)
        self.test_ds = TextDataset(client_data["test_texts"], client_data["test_labels"], tokenizer_name)

    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state_dict = {k: torch.tensor(p).type_as(state_dict[k]) for k, p in zip(state_dict.keys(), parameters)}
        self.model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.local_epochs):
            loop = tqdm(train_loader, desc=f"Client {self.cid} Epoch {epoch+1}", leave=False)
            for batch in loop:
                optimizer.zero_grad()
                inputs = {k:v.to(device) for k,v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

        # Return parameters and also the metrics from a local evaluation
        return self.get_parameters(config), len(self.train_ds), self.evaluate_local()

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # The return format is (loss, num_examples, metrics)
        return 0.0, len(self.test_ds), self.evaluate_local()

    def evaluate_local(self):
        """Evaluate the model on the local test set."""
        self.model.eval()
        loader = DataLoader(self.test_ds, batch_size=self.batch_size)
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device, dtype=torch.long)
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                probs = torch.softmax(outputs.logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        acc = float(accuracy_score(all_labels, all_preds))
        f1 = float(f1_score(all_labels, all_preds, average="macro"))
        recall = float(recall_score(all_labels, all_preds, average="macro"))

        try:
            # Handle case for binary or multiclass AUC
            if len(np.unique(all_labels)) > 2:
                auc = float(roc_auc_score(np.eye(len(np.unique(all_labels)))[all_labels], all_probs, average="macro", multi_class="ovr"))
            else:
                auc = float(roc_auc_score(all_labels, np.array(all_probs)[:, 1]))
        except ValueError:
            auc = 0.0 # Assign a default value if AUC can't be computed

        metrics = {"acc": acc, "f1": f1, "recall": recall, "auc": auc}
        print(f"\n[Client {self.cid}] Metrics: Acc={acc:.4f}, F1={f1:.4f}, Recall={recall:.4f}, AUC={auc:.4f}")
        return metrics

# Run client
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--data_path", type=str, default="dataset.csv")
    args = parser.parse_args()

    # Use the utility function to load and split data
    partitions = load_and_split_csv(args.data_path, args.num_clients)
    client_data = partitions[args.cid]

    # The number of labels for this binary classification task is always 2.
    # Dynamically calculating this from a client's partition can lead to errors
    # if a partition happens to contain only one class.
    num_labels = 2

    # Explicitly load config and set model_type for prajjwal1/bert-tiny, as its config is malformed.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_name)
    config.model_type = 'bert'
    config.num_labels = num_labels

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config
    )

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(args.cid, model, args.model_name, client_data)
    )

