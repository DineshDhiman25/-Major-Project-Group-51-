# Install required packages
!pip install -q "transformers>=4.35" accelerate datasets bitsandbytes peft sentencepiece huggingface_hub sacrebleu rouge-score evaluate
!pip install -q "protobuf<4" --upgrade

# Restart kernel to apply installations
import os
os.kill(os.getpid(), 9)

"""## 3. Import Libraries and Configuration"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import numpy as np
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback
)
import random
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix
)
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

"""## 5. Model and Data Paths Configuration"""

# Model and dataset paths
MODEL_NAME = "/kaggle/input/openhermes25/transformers/default/1"  # Using OpenHermes-2.5-Mistral-7B
DATA_PATH = "/kaggle/input/hrv-finetune/hrv_train.jsonl"
VAL_DATA_PATH = "/kaggle/input/hrv-finetune/hrv_val.jsonl"

# Hyperparameters optimized for P100 16GB
MAX_SEQ_LENGTH = 1536
LANGUAGE_BALANCE = False

# LoRA Configuration - Optimized for P100
LORA_MODE = "attention_only"
LORA_R = 8
LORA_ALPHA = 16

LEARNING_RATE = 2e-5  # Reduced from 1e-4 for stability
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
NUM_EPOCHS = 3

print(f"Model: {MODEL_NAME}")
print(f"Max sequence length: {MAX_SEQ_LENGTH}")
print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")

"""## Focal Loss Trainer"""

import torch
import torch.nn.functional as F
from transformers import Trainer

class FocalLossCausalLMTrainer(Trainer):
    '''
    
    '''
    def __init__(self, *args, focal_gamma=2.0, yes_token_id=None, no_token_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.alpha = torch.tensor([0.15, 0.85])  # Give more weight to "Yes" (violation) class
        self._alpha_device_set = False

        if yes_token_id is None or no_token_id is None:
            raise ValueError("yes_token_id and no_token_id required")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if not self._alpha_device_set:
            self.alpha = self.alpha.to(logits.device)
            self._alpha_device_set = True

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_len, vocab_size = shift_logits.shape

        classification_logits = []
        classification_labels = []

        for b in range(batch_size):
            yes_pos = (shift_labels[b] == self.yes_token_id).nonzero(as_tuple=True)[0]
            no_pos = (shift_labels[b] == self.no_token_id).nonzero(as_tuple=True)[0]

            if len(yes_pos) == 0 and len(no_pos) == 0:
                continue

            all_pos = torch.cat([yes_pos, no_pos])
            pos = all_pos.min().item()
            true_token = shift_labels[b, pos].item()

            yes_logit = shift_logits[b, pos, self.yes_token_id]
            no_logit = shift_logits[b, pos, self.no_token_id]
            binary_logits = torch.stack([no_logit, yes_logit])
            binary_label = 1 if true_token == self.yes_token_id else 0

            classification_logits.append(binary_logits)
            classification_labels.append(binary_label)

        if len(classification_logits) == 0:
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            return (loss, outputs) if return_outputs else loss

        classification_logits = torch.stack(classification_logits)
        classification_labels = torch.tensor(classification_labels, device=logits.device)

        ce_loss = F.cross_entropy(classification_logits, classification_labels, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[classification_labels]
        focal_loss = alpha_t * ((1 - pt) ** self.focal_gamma) * ce_loss

        return (focal_loss.mean(), outputs) if return_outputs else focal_loss.mean()

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import re

def compute_classification_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    pred_tokens = np.argmax(predictions, axis=-1)

    predictions_flat = pred_tokens.flatten()
    labels_flat = labels.flatten()

    mask = labels_flat != -100
    predictions_filtered = predictions_flat[mask]
    labels_filtered = labels_flat[mask]

    accuracy = accuracy_score(labels_filtered, predictions_filtered)

    return {
        "accuracy": accuracy,
        "eval_samples": len(labels),
    }

"""## 5.5 HRV Classification Evaluation Function"""

def evaluate_hrv_classification(model, tokenizer, eval_dataset, device='cuda', max_new_tokens=50):
    """
    Evaluate the model on HRV classification task.
    Generates responses and extracts Yes/No predictions.
    """
    model.eval()
    predictions = []
    ground_truths = []

    print(f"Evaluating on {len(eval_dataset)} samples...")

    for i, sample in enumerate(eval_dataset):
        if i % 20 == 0:
            print(f"  Processing sample {i}/{len(eval_dataset)}")

        messages = sample['messages']

        # Get the ground truth (assistant's response)
        assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), "")
        ground_truth = 1 if assistant_msg.strip().startswith('Yes') else 0
        ground_truths.append(ground_truth)

        # Prepare input (only user message for generation)
        user_messages = [m for m in messages if m['role'] != 'assistant']

        # Format using ChatML template
        prompt = ""
        for msg in user_messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1400)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Extract prediction
        generated_text_lower = generated_text.strip().lower()
        if generated_text_lower.startswith('yes'):
            pred = 1
        elif generated_text_lower.startswith('no'):
            pred = 0
        else:
            # If unclear, default to 0 (no violation)
            pred = 0

        predictions.append(pred)

    # Calculate metrics
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix

    accuracy = accuracy_score(ground_truths, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average='binary', pos_label=1)

    # Detailed report
    report = classification_report(ground_truths, predictions, target_names=['No Violation', 'Violation'], output_dict=True)
    conf_matrix = confusion_matrix(ground_truths, predictions)

    print(f"\n{'='*60}")
    print("HRV Classification Evaluation Results:")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Violation): {precision:.4f}")
    print(f"Recall (Violation): {recall:.4f}")
    print(f"F1-Score (Violation): {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={conf_matrix[0][0]}, FP={conf_matrix[0][1]}")
    print(f"  FN={conf_matrix[1][0]}, TP={conf_matrix[1][1]}")
    print(f"{'='*60}\n")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report,
        'predictions': predictions,
        'ground_truths': ground_truths
    }

"""## 6. Utility Functions"""

def detect_language_improved(text):
    """Detect language based on character composition."""
    if not text:
        return "unknown"
    cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    latin_chars = sum(1 for c in text if ('\u0041' <= c <= '\u005A') or ('\u0061' <= c <= '\u007A'))
    total_chars = cyrillic_chars + latin_chars
    if total_chars == 0:
        return "unknown"
    cyrillic_ratio = cyrillic_chars / total_chars
    if cyrillic_ratio > 0.1:
        return "russian"
    elif latin_chars > 0:
        return "english"
    else:
        return "unknown"


def analyze_translation_artifacts(text):
    """Analyze text for potential translation artifacts."""
    issues = []
    sentences = text.split('.')
    if len(sentences) > 1:
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        if avg_sentence_length > 30:
            issues.append("long_sentences")
    words = text.lower().split()
    if len(words) > 10:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        most_common_count = trigram_counts.most_common(1)[0][1] if trigram_counts else 0
        if most_common_count > 3:
            issues.append("repetitive_ngrams")
    return issues

"""## 7. Dataset Loading and Analysis"""

def load_jsonl_data(file_path, tokenizer, max_samples=None, undersample_ratio=None):
    """Load and analyze JSONL dataset."""
    data = []
    stats = {
        "total": 0,
        "languages": Counter(),
        "truncated": 0,
        "translation_artifacts": Counter(),
        "avg_length": [],
        "user_languages": Counter(),
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break

            entry = json.loads(line.strip())
            messages = entry.get('messages', [])

            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), "")
            assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), "")

            language = detect_language_improved(user_msg)
            stats["user_languages"][language] += 1

            # Use consistent ChatML format (matching tokenize_with_assistant_masking)
            text = ""
            for msg in messages:
                if msg['role'] in ["system", "user", "assistant"]:
                    text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

            token_count = len(tokenizer.encode(text))
            stats["avg_length"].append(token_count)
            if token_count > MAX_SEQ_LENGTH:
                stats["truncated"] += 1

            artifacts = analyze_translation_artifacts(assistant_msg)
            for artifact in artifacts:
                stats["translation_artifacts"][artifact] += 1

            data.append({
                "text": text,
                "messages": messages,
                "language": language,
                "has_artifacts": len(artifacts) > 0
            })

            stats["total"] += 1
            stats["languages"][language] += 1

    print(f"\nDataset Analysis: {file_path}")
    print(f"Total samples: {stats['total']}")
    print(f"User prompt languages: {dict(stats['user_languages'])}")
    print(f"Truncated samples: {stats['truncated']} ({stats['truncated']/stats['total']*100:.1f}%)")
    print(f"Avg token length: {np.mean(stats['avg_length']):.1f}")
    print(f"Translation artifacts: {dict(stats['translation_artifacts'])}")

    if len(stats['user_languages']) >= 2:
        lang_counts = list(stats['user_languages'].values())
        ratio = min(lang_counts) / max(lang_counts)
        print(f"Language balance ratio: {ratio:.2f} (1.0 = perfectly balanced)")
        if ratio > 0.4:
            print("Dataset is already well-balanced!")

    if undersample_ratio is not None:
        yes_data = [d for d in data if d['messages'][1]['content'].startswith('Yes')]
        no_data = [d for d in data if d['messages'][1]['content'].startswith('No')]

        print(f"\nBefore undersampling:")
        print(f"  Yes samples: {len(yes_data)}")
        print(f"  No samples: {len(no_data)}")
        print(f"  Ratio: {len(no_data)/len(yes_data):.2f}:1")

        # Undersample majority class
        target_no_samples = int(len(yes_data) * undersample_ratio)
        if target_no_samples < len(no_data):
            import random
            random.seed(42)  # For reproducibility
            no_data = random.sample(no_data, target_no_samples)

        data = yes_data + no_data
        random.shuffle(data)  # Shuffle combined data

        print(f"\nAfter undersampling:")
        print(f"  Yes samples: {len(yes_data)}")
        print(f"  No samples: {len(no_data)}")
        print(f"  Ratio: {len(no_data)/len(yes_data):.2f}:1")
        print(f"  Total samples: {len(data)}")

        # Update stats
        stats["total"] = len(data)
        stats["class_distribution"] = {
            "yes": len(yes_data),
            "no": len(no_data),
            "ratio": len(no_data)/len(yes_data)
        }

    return data, stats

"""## 8. Tokenization with Assistant Masking"""

def tokenize_with_assistant_masking(examples, tokenizer):
    """
    Tokenize with assistant-only learning.
    Only the assistant responses are used for loss calculation.
    """
    input_ids_list = []
    labels_list = []

    for messages in examples["messages"]:
        full_ids = []
        label_ids = []

        # Manual ChatML-style formatting for OpenHermes
        for message in messages:
            role = message['role']
            content = message['content']

            if role not in ["system", "user", "assistant"]:
                continue

            # OpenHermes uses ChatML format with <|im_start|> and <|im_end|>
            formatted = (
                "<|im_start|>" + role + "\n" +
                content +
                "<|im_end|>\n"
            )

            msg_tokens = tokenizer.encode(formatted, add_special_tokens=False)
            full_ids.extend(msg_tokens)

            if role == 'assistant':
                label_ids.extend(msg_tokens)
            else:
                label_ids.extend([-100] * len(msg_tokens))

        # Add BOS token
        if tokenizer.bos_token_id is not None:
            full_ids = [tokenizer.bos_token_id] + full_ids
            label_ids = [-100] + label_ids

        # Truncate if necessary
        if len(full_ids) > MAX_SEQ_LENGTH:
            full_ids = full_ids[:MAX_SEQ_LENGTH]
            label_ids = label_ids[:MAX_SEQ_LENGTH]

        input_ids_list.append(full_ids)
        labels_list.append(label_ids)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": [[1] * len(ids) for ids in input_ids_list]
    }

"""## 9. Data Collator"""

class DataCollatorForCompletionOnly:
    """Custom data collator that pads sequences correctly."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_length = max(len(f['input_ids']) for f in features)

        input_ids = []
        labels = []
        attention_mask = []

        for feature in features:
            input_id = feature['input_ids']
            label = feature['labels']

            padding_length = max_length - len(input_id)
            input_ids.append(input_id + [self.pad_token_id] * padding_length)
            labels.append(label + [-100] * padding_length)
            attention_mask.append([1] * len(input_id) + [0] * padding_length)

        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'attention_mask': torch.tensor(attention_mask)
        }

"""## 10. Training Monitor Callback"""

# Enhanced Callback for Comprehensive Monitoring
class EnhancedWandbCallback(TrainerCallback):

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.best_eval_loss = float('inf')
        self.steps_without_improvement = 0
        self.gradient_norms = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        '''Log model architecture and LoRA configuration'''
        if model is not None:
            # Count total and trainable parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Create a custom table for LoRA layers
            lora_layers = []
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    lora_layers.append([name, type(module).__name__])

            print(f"\n{'='*60}")
            print(f"Model Parameters Summary:")
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
            print(f"{'='*60}\n")

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        '''Enhanced logging for each step'''
        if logs is not None:
            # Track losses
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])

            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])

                # Calculate overfitting metrics
                if len(self.train_losses) > 0:
                    latest_train_loss = self.train_losses[-1]
                    loss_gap = logs['eval_loss'] - latest_train_loss
                    loss_ratio = logs['eval_loss'] / (latest_train_loss + 1e-10)

                # Track best model
                if logs['eval_loss'] < self.best_eval_loss:
                    self.best_eval_loss = logs['eval_loss']
                    self.steps_without_improvement = 0
                else:
                    self.steps_without_improvement += 1

            # Learning rate tracking
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])

            # Gradient norm tracking
            if 'grad_norm' in logs:
                self.gradient_norms.append(logs['grad_norm'])

            # Memory usage (P100 specific monitoring)
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1e9

    def on_epoch_end(self, args, state, control, **kwargs):
        '''Log epoch-level statistics'''
        if len(self.train_losses) > 0 and len(self.eval_losses) > 0:
            # Calculate moving averages for smoothing
            window = min(10, len(self.train_losses))
            recent_train_losses = self.train_losses[-window:]
            avg_train_loss = np.mean(recent_train_losses)


    def on_train_end(self, args, state, control, **kwargs):
        '''Final summary statistics'''
        if len(self.train_losses) > 0:
            # Overfitting diagnosis
            final_train_loss = np.mean(self.train_losses[-10:])
            final_eval_loss = self.eval_losses[-1] if self.eval_losses else None

            if final_eval_loss:
                diagnosis = "good_fit"
                if final_eval_loss > final_train_loss + 0.5:
                    diagnosis = "overfitting"
                elif final_train_loss > 2.0 and final_eval_loss > 2.0:  # adjust threshold for your task
                    diagnosis = "underfitting"

                print(f"\n{'='*60}")
                print(f"Training Diagnosis: {diagnosis.upper()}")
                print(f"Final Train Loss: {final_train_loss:.4f}")
                print(f"Final Eval Loss: {final_eval_loss:.4f}")
                print(f"Best Eval Loss: {self.best_eval_loss:.4f}")
                print(f"{'='*60}\n")


# Optional: Per-Layer Gradient Monitoring (can slow training)
class GradientMonitorCallback(TrainerCallback):
    '''
    Optional: Monitor gradient norms per layer for detailed debugging
    WARNING: This can slow down training significantly
    '''

    def __init__(self, log_frequency=100):
        self.log_frequency = log_frequency

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and state.global_step % self.log_frequency == 0:
            gradient_data = []

            for name, param in model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    grad_norm = param.grad.norm().item()
                    gradient_data.append([name, grad_norm, param.numel()])


"""## 11. LoRA Configuration Helper"""

def get_lora_target_modules(mode="attention_only"):
    """Get LoRA target modules based on strategy."""
    configs = {
        "attention_only": {
            "modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "description": "Conservative: Attention layers only (RECOMMENDED for bilingual data)"
        },
        "attention_mlp": {
            "modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "description": "Moderate: Attention + MLP (use if underfitting)"
        }
    }

    config = configs.get(mode, configs["attention_only"])
    print(f"\nLoRA Mode: {mode}")
    print(f"   {config['description']}")
    print(f"   Target modules: {config['modules']}")

    return config['modules']

"""## 12. Main Training Function"""


print("Loading tokenizer...")
sys.stdout.flush()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("\nLoading datasets...")
sys.stdout.flush()
train_data, train_stats = load_jsonl_data(
    DATA_PATH,
    tokenizer,
    undersample_ratio=1.5  # Target 1.5:1 ratio for better recall on violations
)
val_data, val_stats = load_jsonl_data(
    VAL_DATA_PATH,
    tokenizer,
    max_samples=500,
    undersample_ratio=None  # Don't undersample validation set
)

# Log dataset distribution
if "user_languages" in train_stats:
    lang_data = [[lang, count] for lang, count in train_stats["user_languages"].items()]

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print("\nTokenizing with assistant-only masking...")
sys.stdout.flush()
train_tokenized = train_dataset.map(
    lambda x: tokenize_with_assistant_masking(x, tokenizer),
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train"
)

val_tokenized = val_dataset.map(
    lambda x: tokenize_with_assistant_masking(x, tokenizer),
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing val"
)

print("\nLoading OpenHermes model with 4-bit quantization (P100 optimized)...")
sys.stdout.flush()

# P100 doesn't support bfloat16, use float16 instead
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # Changed from bfloat16 to float16 for P100
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16,  # Changed from bfloat16 to float16 for P100
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
if hasattr(model.config, "pretraining_tp"):
    model.config.pretraining_tp = 1

print("\nConfiguring LoRA...")
sys.stdout.flush()
target_modules = get_lora_target_modules(LORA_MODE)

effective_lora_r = min(LORA_R, 8)
effective_lora_alpha = min(LORA_ALPHA, 16)

peft_config = LoraConfig(
    r=effective_lora_r,
    lora_alpha=effective_lora_alpha,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention only
    lora_dropout=0.15,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Get token IDs for focal loss with verification
yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
no_tokens = tokenizer.encode("No", add_special_tokens=False)

# Use the first token, with fallback to tokens with leading space
yes_token_id = yes_tokens[0] if yes_tokens else tokenizer.encode(" Yes", add_special_tokens=False)[0]
no_token_id = no_tokens[0] if no_tokens else tokenizer.encode(" No", add_special_tokens=False)[0]

# Verify tokens decode correctly
yes_decoded = tokenizer.decode([yes_token_id])
no_decoded = tokenizer.decode([no_token_id])
print(f"\nToken IDs - Yes: {yes_token_id} (decodes to: '{yes_decoded}')")
print(f"Token IDs - No: {no_token_id} (decodes to: '{no_decoded}')")

# Sanity check
if 'yes' not in yes_decoded.lower() or 'no' not in no_decoded.lower():
    print("WARNING: Token IDs may not correctly represent Yes/No. Check tokenizer behavior.")

# Training args WITHOUT early_stopping_patience

"""## 13. Execute Training"""

training_args = TrainingArguments(
    output_dir="/kaggle/working/openhermes-finetuned",
    run_name="openhermes-hrv-finetuning",
    num_train_epochs=4,  # Increased from 2 for better convergence
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=5e-6,
    weight_decay=0.05,
    max_grad_norm=0.3,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    fp16=True,
    tf32=False,
    logging_steps=5,
    logging_first_step=True,
    eval_strategy="steps",
    eval_steps=30,
    save_strategy="steps",
    save_steps=30,
    save_total_limit=6,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    group_by_length=True,
    disable_tqdm=False,
    log_level="info",
)

data_collator = DataCollatorForCompletionOnly(tokenizer)

print("\nInitializing Trainer with enhanced callbacks...")
sys.stdout.flush()

enhanced_callback = EnhancedWandbCallback()

trainer = FocalLossCausalLMTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    callbacks=[enhanced_callback],
    focal_gamma=2.0,  # Reduced from 2.5 for less conservative predictions
    yes_token_id=yes_token_id,
    no_token_id=no_token_id,
)

# Checkpoint detection
checkpoint_dir = "/kaggle/working/openhermes-finetuned"
last_checkpoint = None

if os.path.exists(checkpoint_dir):
    checkpoints = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        last_checkpoint = checkpoints[-1]
        checkpoint_step = last_checkpoint.split("-")[-1]
        print(f"\nFound checkpoint at step {checkpoint_step}")
        print(f"Resuming from: {last_checkpoint}")

if last_checkpoint is None:
    print("\nStarting training from scratch...")
else:
    print(f"\nResuming training from checkpoint...")

print("=" * 60)
sys.stdout.flush()
torch.cuda.empty_cache()

# Start training
trainer.train(resume_from_checkpoint=last_checkpoint)

print("\n" + "=" * 60)
print("Saving final model...")
sys.stdout.flush()
final_dir = "/kaggle/working/openhermes-finetuned-final"
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)


print("\nTraining completed successfully!")
sys.stdout.flush()

print("\n" + "="*60)
print("Running comprehensive evaluation on validation set...")
print("="*60)

# Evaluate
best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    print(f"Loading best model from: {best_model_path}")

try:
    eval_results = evaluate_hrv_classification(
        model=trainer.model,
        tokenizer=tokenizer,
        eval_dataset=val_data[:100],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
except Exception as e:
    print(f"Warning: Comprehensive evaluation failed: {e}")
    print("Continuing with standard metrics only.")

# Print training summary (removed return statement as it's not inside a function)
training_summary = {
    "final_train_loss": enhanced_callback.train_losses[-1] if enhanced_callback.train_losses else None,
    "final_eval_loss": enhanced_callback.eval_losses[-1] if enhanced_callback.eval_losses else None,
    "best_eval_loss": enhanced_callback.best_eval_loss,
    "train_samples": train_stats["total"],
    "val_samples": val_stats["total"],
    "train_languages": dict(train_stats["user_languages"]),
    "resumed_from_checkpoint": last_checkpoint is not None,
    "checkpoint_path": last_checkpoint if last_checkpoint else "none",
    "config": {
        "model": MODEL_NAME,
        "lora_mode": LORA_MODE,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "max_seq_length": MAX_SEQ_LENGTH,
        "precision": "fp16",
        "gpu": "P100"
    }
}

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
for key, value in training_summary.items():
    print(f"{key}: {value}")
print("="*60)

!zip -r hermes_final.zip /kaggle/working/openhermes-finetuned-final
