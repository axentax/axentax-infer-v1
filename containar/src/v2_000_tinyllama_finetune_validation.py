#!/usr/bin/env python3
"""
TinyLlama 1.1B QLoRA ファインチューニング（Validation Loss監視版）

改善点:
- validation.jsonlを使用したvalidation loss監視
- 過学習検出のための早期停止
- より詳細な評価メトリクス
- 学習曲線の記録
"""

import json
import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np


def load_jsonl_data(file_paths):
    """Load and combine multiple JSONL files"""
    all_data = []
    for path in file_paths:
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # TinyLlama chat format
                input_text = item.get('input', '').strip()
                if input_text:
                    text = f"<|user|>\n{item['instruction']}\n{input_text}<|end|>\n<|assistant|>\n{item['output']}<|end|>"
                else:
                    text = f"<|user|>\n{item['instruction']}<|end|>\n<|assistant|>\n{item['output']}<|end|>"
                all_data.append({"text": text, "meta": item.get("meta", {})})
    return all_data


def compute_metrics(eval_pred):
    """カスタム評価メトリクス"""
    predictions, labels = eval_pred
    # 単純にvalidation lossを返す（より複雑な評価は別途実装）
    return {}


def main():
    # Configuration
    model_path = "./tmp_docker/models/tinyllama-1.1b-chat"
    train_paths = ["./tmp_docker/dataset/train.jsonl", "./tmp_docker/dataset/axentax_full.jsonl"]
    val_path = "./tmp_docker/dataset/validation.jsonl"
    output_dir = "./src/output/v2_with_validation"
    
    # より保守的な学習設定
    max_steps = 500  # validation監視のために増加
    eval_steps = 50   # 頻繁な評価
    save_steps = 100
    
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
    model.config.use_cache = False
    
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
    
    print("Loading validation data...")
    val_data = load_jsonl_data([val_path])
    print(f"Loaded {len(val_data)} validation examples")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Tokenize function
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        result["labels"] = result["input_ids"].clone()
        return result
    
    # Apply tokenization
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "meta"]
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "meta"]
    )
    
    # Training arguments with validation
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        learning_rate=2e-4,  # 少し低めに設定
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
        logging_dir=f"{output_dir}/logs"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )
    
    # Trainer with validation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )
    
    # Train
    print(f"Starting training with validation monitoring...")
    print(f"Output: {output_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    try:
        # 初回評価
        print("Initial evaluation...")
        initial_eval = trainer.evaluate()
        print(f"Initial validation loss: {initial_eval['eval_loss']:.4f}")
        
        # 学習実行
        trainer.train()
        
        # 最終評価
        print("Final evaluation...")
        final_eval = trainer.evaluate()
        print(f"Final validation loss: {final_eval['eval_loss']:.4f}")
        
        # Save final model
        print("Saving final model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # 学習履歴の保存
        with open(f"{output_dir}/training_history.json", "w") as f:
            json.dump({
                "initial_eval_loss": initial_eval['eval_loss'],
                "final_eval_loss": final_eval['eval_loss'],
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "config": {
                    "max_steps": max_steps,
                    "eval_steps": eval_steps,
                    "learning_rate": 2e-4,
                    "lora_r": 16,
                    "lora_alpha": 32
                }
            }, indent=2)
        
        print(f"Training completed! Model saved to: {output_dir}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save what we can
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Partial model saved to: {output_dir}")
        except:
            print("Could not save model")


if __name__ == "__main__":
    main()