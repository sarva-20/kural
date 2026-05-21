import json
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

MAX_SEQ_LENGTH = 2048
TEST_CHUNKS = [
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company is headquartered in Cupertino, California. Apple designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
    
    "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints. It was created by Sebastián Ramírez and first released in 2018. FastAPI is built on top of Starlette and Pydantic.",
    
    "The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It uses self-attention mechanisms to process sequential data. Transformers have become the dominant architecture for natural language processing tasks.",
]

INSTRUCTION = "Extract structured information from this web content chunk as JSON with fields: title, summary, links, entities."

def run_inference(model, tokenizer, chunk: str) -> str:
    FastLanguageModel.for_inference(model)

    messages = [
        {
            "role": "system",
            "content": "You are a precise information extraction system. Given a raw web content chunk, extract structured information as JSON. Output ONLY valid JSON."
        },
        {
            "role": "user",
            "content": f"{INSTRUCTION}\n\n{chunk}"
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
    )

    return tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True
    )


def is_valid_json(text: str) -> bool:
    try:
        import re
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text).strip()
        parsed = json.loads(text)
        required = {"title", "summary", "links", "entities"}
        return required.issubset(parsed.keys())
    except:
        return False


# ── Load BASE model ───────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Kural Evaluation — Base vs Fine-tuned")
print("="*60)

print("\n[1/2] Loading BASE Llama 3.2 3B Instruct...")
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

print("\n--- BASE MODEL RESULTS ---")
base_results = []
for i, chunk in enumerate(TEST_CHUNKS, 1):
    print(f"\nTest {i}: {chunk[:60]}...")
    response = run_inference(base_model, tokenizer, chunk)
    valid = is_valid_json(response)
    base_results.append(valid)
    print(f"Output: {response[:200]}")
    print(f"Valid JSON: {'✅' if valid else '❌'}")

# ── Load FINE-TUNED model ─────────────────────────────────────────────────────

print("\n[2/2] Loading FINE-TUNED Kural adapter...")
ft_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
ft_model = PeftModel.from_pretrained(ft_model, "kural-adapter")

print("\n--- KURAL FINE-TUNED RESULTS ---")
ft_results = []
for i, chunk in enumerate(TEST_CHUNKS, 1):
    print(f"\nTest {i}: {chunk[:60]}...")
    response = run_inference(ft_model, tokenizer, chunk)
    valid = is_valid_json(response)
    ft_results.append(valid)
    print(f"Output: {response[:200]}")
    print(f"Valid JSON: {'✅' if valid else '❌'}")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  EVALUATION SUMMARY")
print(f"{'='*60}")
print(f"  Base Llama 3.2 3B  — Valid JSON: {sum(base_results)}/{len(base_results)}")
print(f"  Kural Fine-tuned   — Valid JSON: {sum(ft_results)}/{len(ft_results)}")
print(f"{'='*60}\n")