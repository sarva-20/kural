import json
import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_FILE = "dataset.jsonl"
OUTPUT_DIR = "kural-adapter"
MAX_SEQ_LENGTH = 2048

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10

# ── Load dataset ──────────────────────────────────────────────────────────────

def load_dataset_from_jsonl(path: str) -> Dataset:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"Loaded {len(samples)} samples from {path}")
    return Dataset.from_list(samples)


def format_sample(sample: dict) -> str:
    """
    Converts Alpaca format to Llama 3.2 chat template format.
    This is what the model actually trains on.
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise information extraction system. Given a raw web content chunk, extract structured information as JSON. Output ONLY valid JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>
{sample['instruction']}

{sample['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{sample['output']}<|eot_id|>"""


# ── Load model ────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Kural Training Run")
print(f"  Model: {MODEL_NAME}")
print(f"  Dataset: {DATASET_FILE}")
print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
print(f"  Epochs: {EPOCHS}")
print("="*60 + "\n")

print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# ── Apply LoRA ────────────────────────────────────────────────────────────────

print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Print trainable parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")

# ── Prepare dataset ───────────────────────────────────────────────────────────

print("\nPreparing dataset...")
raw_dataset = load_dataset_from_jsonl(DATASET_FILE)

# Format all samples
def preprocess(sample):
    return {"text": format_sample(sample)}

dataset = raw_dataset.map(preprocess)

# 90/10 train/eval split
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples:  {len(eval_dataset)}")

# ── Train ─────────────────────────────────────────────────────────────────────

print("\nStarting training...\n")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    ),
)

trainer_stats = trainer.train()

# ── Save adapter ──────────────────────────────────────────────────────────────

print("\nSaving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n{'='*60}")
print(f"  Training Complete")
print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.0f}s ({trainer_stats.metrics['train_runtime']/60:.1f} mins)")
print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")
print(f"  Adapter saved to: {OUTPUT_DIR}/")
print(f"{'='*60}\n")