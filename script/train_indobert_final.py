import os
import shutil
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from scipy.special import softmax
import wandb
import uuid

# --- Config ---
BASE_DIR = "./"
MODEL_NAME = "indobenchmark/indobert-base-p1"
DATA_PATH = os.path.join(BASE_DIR, "cleandata/train_dataset_cleaned.csv")
RESULT_DIR = os.path.join(BASE_DIR, "results_indobert")
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")
NUM_FOLDS = 5
BATCH_SIZE = 16
EPOCHS = 10
PATIENCE = 3
SEED = 42
TRUNCATION_LENGTH = 480
WANDB_PROJECT = "IndoBERT-Hoax-Detection-Colab"

# --- Setup Directories ---
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Full Reset ---
CLEAR_PREVIOUS = True

if CLEAR_PREVIOUS:
    if os.path.exists(RESULT_DIR):
        print(f"Clearing previous results at '{RESULT_DIR}'...")
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR, exist_ok=True)

    if os.path.exists("wandb"):
        print("Clearing local Weights & Biases logs...")
        shutil.rmtree("wandb")

# --- Init Weights & Biases ---
run_id = str(uuid.uuid4())[:8]
wandb.init(project=WANDB_PROJECT, name=f"IndoBERT-5Fold-{run_id}", reinit=True)

# --- Load Dataset ---
def load_data(data_path):
    df = pd.read_csv(data_path).dropna(subset=["Text", "label"])
    df["label"] = df["label"].astype(int)
    return df

df = load_data(DATA_PATH)

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["Text"], truncation=True, padding=True, max_length=TRUNCATION_LENGTH, return_length=True)

# --- Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Stratified K-Fold Setup ---
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
all_metrics = []

# --- Training Function ---
def train_model(train_df, val_df, fold):
    output_dir = os.path.join(RESULT_DIR, f"fold_{fold+1}")
    os.makedirs(output_dir, exist_ok=True)
    model_save_dir = os.path.join(SAVE_DIR, f"fold_{fold+1}")
    os.makedirs(model_save_dir, exist_ok=True)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        logging_dir=os.path.join(output_dir, "logs"),
        save_total_limit=1,
        seed=SEED,
        report_to="wandb",
        fp16=torch.cuda.is_available()
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    trainer.train()
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    return trainer, val_dataset

# --- Evaluation Function ---
def evaluate_model(trainer, val_dataset, fold):
    preds = trainer.predict(val_dataset)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    probs = softmax(preds.predictions, axis=1)

    pd.Series(np.max(probs, axis=1)).describe().to_csv(os.path.join(RESULT_DIR, f"fold_{fold+1}", "confidence_stats.csv"))

    pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose().to_csv(
        os.path.join(RESULT_DIR, f"fold_{fold+1}", "classification_report.csv"))
    np.savetxt(os.path.join(RESULT_DIR, f"fold_{fold+1}", "confusion_matrix.txt"), confusion_matrix(y_true, y_pred), fmt="%d")

    return y_true, y_pred, probs

# --- Plotting Functions ---
def plot_confusion_matrix(y_true, y_pred, fold):
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Valid", "Hoax"], yticklabels=["Valid", "Hoax"])
    plt.title(f"Confusion Matrix - Fold {fold+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"fold_{fold+1}", "confusion_matrix.png"))
    plt.close()

def plot_roc_curve(y_true, probs, fold):
    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, probs[:, 1]):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Fold {fold+1}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"fold_{fold+1}", "roc_curve.png"))
    plt.close()

def plot_loss_curve(trainer, fold):
    train_loss = [(log["epoch"], log["loss"]) for log in trainer.state.log_history if "loss" in log and "eval_loss" not in log]
    val_loss = [(log["epoch"], log["eval_loss"]) for log in trainer.state.log_history if "eval_loss" in log]
    if train_loss:
        train_epochs, train_losses = zip(*train_loss)
        val_epochs, val_losses = zip(*val_loss)
        plt.figure()
        plt.plot(train_epochs, train_losses, label="Train Loss", color="blue")
        plt.plot(val_epochs, val_losses, label="Val Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve - Fold {fold+1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"fold_{fold+1}", "loss_curve.png"))
        plt.close()

# --- Training per fold ---
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    print(f"Fold {fold+1}/{NUM_FOLDS}")
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    trainer, val_dataset = train_model(train_df, val_df, fold)
    y_true, y_pred, probs = evaluate_model(trainer, val_dataset, fold)

    plot_confusion_matrix(y_true, y_pred, fold)
    plot_roc_curve(y_true, probs, fold)
    plot_loss_curve(trainer, fold)

    # --- Log Fold Metrics to W&B ---
    wandb.log({
        f"fold_{fold+1}/accuracy": accuracy_score(y_true, y_pred),
        f"fold_{fold+1}/f1": f1_score(y_true, y_pred),
        f"fold_{fold+1}/roc_auc": roc_auc_score(y_true, probs[:, 1]),
    })

    all_metrics.append({
        "fold": fold + 1,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, probs[:, 1]),
    })

# --- Save Summary ---
summary_df = pd.DataFrame(all_metrics)
summary_df.to_csv(os.path.join(RESULT_DIR, "summary_metrics.csv"), index=False)
print("\n✅ Training complete. Results saved to:", RESULT_DIR)

wandb.finish()


