#!/usr/bin/env python3
"""
Gemma-3 270M LoRAファインチューニングスクリプト

実行方法:
  python src/v1_lora_finetune_000.py

設定:
  - モデル: Gemma-3 270M (./tmp_docker/models/gemma-3-270m-it)
  - データセット: Axentax (train.jsonl + axentax_full.jsonl = 28,553例)
  - 学習: 100ステップ (テスト用)
  - GPU使用 + LoRA (r=16, alpha=32)
  - 保存先: ./src/output/v1

動作確認済み:
  - 正常学習完了（100ステップ）
  - 50ステップ・100ステップでモデル保存
  - LoRAアダプター15MB保存完了
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os


def load_jsonl_data(file_paths):
    """Load and combine multiple JSONL files"""
    all_data = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # Simple format for instruction following
                text = f"[INST] {item['instruction']}\n{item['input']} [/INST] {item['output']}"
                all_data.append({"text": text})
    return all_data


def main():
    # Configuration - CHANGED OUTPUT PATH
    model_path = "./tmp_docker/models/gemma-3-270m-it"
    train_paths = ["./tmp_docker/dataset/train.jsonl", "./tmp_docker/dataset/axentax_full.jsonl"]
    output_dir = "./src/output/v1"  # CHANGED TO ./src/output/v1 (versioned)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.config.use_cache = False  # For training
    
    # LoRA configuration
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print("Loading training data...")
    train_data = load_jsonl_data(train_paths)
    print(f"Loaded {len(train_data)} training examples")
    
    # Create dataset
    train_dataset = Dataset.from_list(train_data)
    
    # Tokenize function
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        # For causal LM, labels are the same as input_ids
        result["labels"] = result["input_ids"].clone()
        return result
    
    # Apply tokenization
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=100,  # Short test
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        learning_rate=5e-4,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=50,  # Save at step 50 and 100
        save_strategy="steps",
        save_total_limit=2,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"Starting training... Output will be saved to: {output_dir}")
    try:
        trainer.train()
        
        # Save final model
        print("Saving final model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Training completed! Model saved to: {output_dir}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        # Save what we can
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Partial model saved to: {output_dir}")
        except:
            print("Could not save model")


if __name__ == "__main__":
    main()