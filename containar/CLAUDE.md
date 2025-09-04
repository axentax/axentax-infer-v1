## 目的
データセットを使用してGemma-3 270Mをファインチューニングする。まずは1エポック
ディスク容量が無駄になるので学習の途中ステップは保存しなくて良い。

## ファインチューニング詳細
LoRAを使用する

## ソースにバージョン接頭辞を付与する。改変した場合は3桁番号をインクリメント
> 命名規則 v{VERSION}_{3桁番号}_{機能名}.py に統一
v1_000_lora_finetune.py
v1_001_lora_finetune_simple.py
v1_000_infer.py
など

## 置き場所
- チューニング済みのモデルの置き場所\
./src/output

- ソースコードの置き場所
./src

- 元モデル\
tmp_docker/models

- データセット
tmp_docker/dataset

## 節約
学習はGPUを使用
推論はGPUを使用しない

## メモリエラー対処方法
CLAUDE_memory_error.md

## todo
TinyLlama 1.1B

### 学習の目安・ハイパラ
エポック: データ質次第。まず 1〜3 epoch、Valで早期停止検討。
LoRA r: 16→32に上げると表現力↑ただしメモリ↑
LR: 5e-5〜2e-4 をスイープ
max_seq_length: 1024/2048 から開始（長文多ければ2048）
packing: SFTは packing=True が効果的（短サンプル多いほど）

### 学習スクリプトサンプル
```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinyLlama 1.1B QLoRA SFT
- 4bit QLoRA(NF4)
- TRL SFTTrainer
- instruction/input/output と messages の両対応
"""

import os, json
from dataclasses import dataclass
from typing import Dict, List, Any
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_PATH  = os.environ.get("DATA_PATH", "./dataset/train.jsonl")  # JSONL
VAL_PATH   = os.environ.get("VAL_PATH", "./dataset/validation.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output_tinyllama_lora")

MAX_LEN    = int(os.environ.get("MAX_LEN", "2048"))
MICRO_BSZ  = int(os.environ.get("MICRO_BSZ", "1"))
GR_ACC     = int(os.environ.get("GR_ACC", "16"))
LR         = float(os.environ.get("LR", "1e-4"))
EPOCHS     = float(os.environ.get("EPOCHS", "3"))
WARMUP     = float(os.environ.get("WARMUP", "0.03"))

def detect_schema(example: Dict[str, Any]) -> str:
    if "messages" in example:
        return "chat"
    elif all(k in example for k in ["instruction","output"]):
        return "instruct"
    return "unknown"

def format_example(example: Dict[str, Any]) -> str:
    schema = detect_schema(example)
    if schema == "chat":
        # ChatML風を素直に連結（シンプル版）
        parts = []
        for m in example["messages"]:
            role = m["role"]
            content = m["content"]
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|end|>")
        return "\n".join(parts)

    if schema == "instruct":
        instr = example.get("instruction","").strip()
        inp   = example.get("input","").strip()
        out   = example.get("output","").strip()
        if inp:
            prompt = f"<|system|>\nあなたは有能なアシスタントです。\n<|user|>\n{instr}\n{inp}\n<|assistant|>\n{out}"
        else:
            prompt = f"<|system|>\nあなたは有能なアシスタントです。\n<|user|>\n{instr}\n<|assistant|>\n{out}"
        return prompt

    # スキップ（Trainer側でremove_unused_columns=Falseなので無害）
    return ""

def formatting_func(examples: Dict[str, List[Any]]) -> List[str]:
    texts = []
    for i in range(len(examples[next(iter(examples))])):
        item = {k: examples[k][i] for k in examples}
        txt = format_example(item)
        if txt:
            texts.append(txt)
    return texts

def main():
    # 4bit 量子化（QLoRA）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # 新規pad_tokenを与えた場合のresize
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],  # LLama系一般
        bias="none",
        task_type="CAUSAL_LM"
    )

    # データロード（jsonl）
    train_ds = load_dataset("json", data_files=DATA_PATH, split="train")
    val_ds   = load_dataset("json", data_files=VAL_PATH, split="train") if os.path.exists(VAL_PATH) else None

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=MICRO_BSZ,
        gradient_accumulation_steps=GR_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps" if val_ds else "no",
        eval_steps=500 if val_ds else None,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=lora,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        max_seq_length=MAX_LEN,
        packing=True,  # 連結詰めで高速化
        args=args
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)   # adapter を含む
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
```
