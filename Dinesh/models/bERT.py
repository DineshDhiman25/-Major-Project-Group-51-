# ============================================================
# 🚀 FAST HRV DETECTOR TRAINING — Optimized for Colab GPU
# ============================================================

import sys, torch, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ---- Safety: disable any unwanted W&B logging ----
sys.modules["wandb"] = None

# ---- Torch speed tweaks ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ============================================================
# 1️⃣  LOAD & PREPARE DATA
# ============================================================
hrv  = pd.read_csv("hrv.csv")[["Sr. no.", "Post", "HRV Result"]]
nhrv = pd.read_csv("nhrv.csv")[["Sr. no.", "Post", "HRV Result"]]

combined = pd.concat([hrv, nhrv], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
train_df, test_df = train_test_split(combined, test_size=0.2, random_state=42)

def normalize_label(x):
    return 1 if str(x).strip().lower() in ["1", "hrv", "yes", "true"] else 0

train_df["HRV Result"] = train_df["HRV Result"].apply(normalize_label)
test_df["HRV Result"]  = test_df["HRV Result"].apply(normalize_label)

print(f"✅ Train: {train_df.shape}, Test: {test_df.shape}")

# ============================================================
# 2️⃣  TOKENIZER & DATASETS  (shorter seq_len = faster)
# ============================================================
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # ✅ fast, public, multilingual
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["Post"], padding="max_length", truncation=True, max_length=48)

train_ds = HFDataset.from_pandas(train_df[["Post","HRV Result"]]).map(tokenize, batched=True)
test_ds  = HFDataset.from_pandas(test_df[["Post","HRV Result"]]).map(tokenize, batched=True)

train_ds = train_ds.rename_column("HRV Result","labels")
test_ds  = test_ds.rename_column("HRV Result","labels")
train_ds.set_format("torch", columns=["input_ids","attention_mask","labels"])
test_ds.set_format("torch", columns=["input_ids","attention_mask","labels"])

# ============================================================
# 3️⃣  MODEL
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.gradient_checkpointing_enable()   # saves memory

# ============================================================
# 4️⃣  TRAINING ARGUMENTS (fast mode)
# ============================================================
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,   # bigger batch → faster
    per_device_eval_batch_size=32,
    num_train_epochs=1,               # one epoch fine for small data
    weight_decay=0.01,
    fp16=True,                        # mixed precision
    logging_dir="./logs",
    load_best_model_at_end=True,
    report_to=[],                     # disable online logging
)

# ============================================================
# 5️⃣  TRAINER
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ============================================================
# 6️⃣  TRAIN 🚀
# ============================================================
trainer.train()

# ============================================================
# 7️⃣  EVALUATE & SAVE
# ============================================================
print("\n✅ Evaluating on test set...")
preds_output = trainer.predict(test_ds)
preds = preds_output.predictions.argmax(-1)
labels = preds_output.label_ids

print("\nClassification Report:\n", classification_report(labels, preds, digits=4))
print("Confusion Matrix:\n", confusion_matrix(labels, preds))

trainer.save_model("hrv_model_fast")
tokenizer.save_pretrained("hrv_model_fast")
print("\n✅ Fast model saved as 'hrv_model_fast/'")
