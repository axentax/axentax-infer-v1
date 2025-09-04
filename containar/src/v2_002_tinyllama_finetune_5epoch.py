#!/usr/bin/env python3
"""
TinyLlama 1.1B 5エポック学習（Validation監視付き）

設定:
- 5エポック学習
- validation.jsonl使用
- 早期停止機能付き
- GPU使用
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


def main():
    # Configuration
    model_path = "./tmp_docker/models/tinyllama-1.1b-chat"
    train_paths = ["./tmp_docker/dataset/train.jsonl", "./tmp_docker/dataset/axentax_full.jsonl"]
    val_path = "./tmp_docker/dataset/validation.jsonl"
    output_dir = "./src/output/v2_5epoch"
    
    # 5エポック設定
    num_epochs = 5
    eval_steps = 500   # validation評価間隔
    save_steps = 1000  # 保存間隔
    
    print(f"🎯 TinyLlama 1.1B - {num_epochs}エポック学習開始")
    print("=" * 60)
    
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
        attn_implementation="eager",
        device_map="auto"
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
    
    # Training arguments for 5 epochs
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        learning_rate=1e-4,  # 5エポックなので少し控えめ
        bf16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
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
    
    # Early stopping callback (patience=3 for 5 epochs)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,  # 5エポックなので少し長めに
        early_stopping_threshold=0.01
    )
    
    # Trainer with validation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping]
    )
    
    # Calculate total steps
    total_steps = len(train_dataset) * num_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    
    print(f"🚀 学習開始:")
    print(f"  - エポック数: {num_epochs}")
    print(f"  - 学習データ: {len(train_dataset)}")
    print(f"  - 検証データ: {len(val_dataset)}")
    print(f"  - 予想総ステップ数: {total_steps}")
    print(f"  - 出力先: {output_dir}")
    
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
        with open(f"{output_dir}/training_summary.json", "w") as f:
            json.dump({
                "epochs": num_epochs,
                "initial_eval_loss": initial_eval['eval_loss'],
                "final_eval_loss": final_eval['eval_loss'],
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "total_steps_completed": trainer.state.global_step,
                "config": {
                    "learning_rate": 1e-4,
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "max_length": 512
                }
            }, indent=2)
        
        print(f"✅ {num_epochs}エポック学習完了! モデル保存先: {output_dir}")
        
    except Exception as e:
        print(f"❌ 学習中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        
        # Save what we can
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"部分的にモデルを保存: {output_dir}")
        except:
            print("モデル保存に失敗")


if __name__ == "__main__":
    main()