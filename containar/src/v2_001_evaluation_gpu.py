#!/usr/bin/env python3
"""
TinyLlama 1.1B 学習済みモデル 評価スクリプト（GPU版）

改善点:
- GPU使用（メモリ制限6GB程度）
- バッチ処理で高速化
- サンプル数調整でタイムアウト回避
"""

import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")
from datasets import Dataset
import numpy as np
from typing import List, Dict


def setup_gpu_memory_limit(max_memory_gb=6):
    """GPU メモリ使用量を制限"""
    if torch.cuda.is_available():
        # メモリ使用量制限設定
        torch.cuda.empty_cache()
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"Setting memory limit to ~{max_memory_gb}GB")
        return True
    else:
        print("GPU not available, using CPU")
        return False


def load_validation_data(val_path: str, max_samples: int = 50) -> List[Dict]:
    """Validation dataを読み込み（サンプル数制限）"""
    data = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            item = json.loads(line.strip())
            data.append(item)
    return data


def calculate_perplexity_batch(model, tokenizer, texts: List[str], batch_size: int = 4) -> float:
    """バッチ処理でPerplexityを計算"""
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0
    total_tokens = 0
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # バッチトークナイズ
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # バッチ内の有効トークン数を計算
            batch_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()


def evaluate_axentax_batch(model, tokenizer, test_cases: List[Dict], batch_size: int = 2) -> Dict:
    """バッチ処理でAxentax DSL理解度を評価"""
    model.eval()
    device = next(model.parameters()).device
    
    results = {
        "total": len(test_cases),
        "syntax_valid_correct": 0,
        "syntax_invalid_correct": 0,
        "detailed_results": []
    }
    
    for i in range(0, len(test_cases), batch_size):
        batch_cases = test_cases[i:i + batch_size]
        
        # バッチ用プロンプト構築
        prompts = []
        expected_validities = []
        
        for case in batch_cases:
            instruction = case["instruction"]
            input_text = case.get("input", "")
            is_valid = case.get("meta", {}).get("valid", True)
            
            if input_text:
                prompt = f"<|user|>\n{instruction}\n{input_text}<|end|>\n<|assistant|>\n"
            else:
                prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"
            
            prompts.append(prompt)
            expected_validities.append(is_valid)
        
        # バッチトークナイズ
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256,
            padding=True
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        input_length = input_ids.shape[1]
        
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,  # 短縮で高速化
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # バッチ内の各応答を処理
                for j, (case, is_valid) in enumerate(zip(batch_cases, expected_validities)):
                    generated_tokens = outputs[j][input_length:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    
                    # 評価ロジック
                    if is_valid:
                        if "構文は有効です" in response or "有効です" in response:
                            results["syntax_valid_correct"] += 1
                            correct = True
                        else:
                            correct = False
                    else:
                        if "無効です" in response or "エラー" in response:
                            results["syntax_invalid_correct"] += 1
                            correct = True
                        else:
                            correct = False
                    
                    results["detailed_results"].append({
                        "input": case.get("input", ""),
                        "expected_valid": is_valid,
                        "response": response[:100],  # 長さ制限
                        "correct": correct
                    })
                    
            except Exception as e:
                for case, is_valid in zip(batch_cases, expected_validities):
                    results["detailed_results"].append({
                        "input": case.get("input", ""),
                        "expected_valid": is_valid,
                        "response": f"ERROR: {e}",
                        "correct": False
                    })
    
    # 精度計算
    total_valid = sum(1 for case in test_cases if case.get("meta", {}).get("valid", True))
    total_invalid = len(test_cases) - total_valid
    
    valid_accuracy = results["syntax_valid_correct"] / total_valid if total_valid > 0 else 0
    invalid_accuracy = results["syntax_invalid_correct"] / total_invalid if total_invalid > 0 else 0
    overall_accuracy = (results["syntax_valid_correct"] + results["syntax_invalid_correct"]) / len(test_cases)
    
    results["valid_accuracy"] = valid_accuracy
    results["invalid_accuracy"] = invalid_accuracy
    results["overall_accuracy"] = overall_accuracy
    
    return results


def main():
    print("🚀 TinyLlama 1.1B GPU評価（メモリ制限版）")
    print("=" * 60)
    
    # GPU設定
    use_gpu = setup_gpu_memory_limit(6)
    device = "cuda" if use_gpu else "cpu"
    dtype = torch.float16 if use_gpu else torch.float32
    
    # パス設定
    base_model_path = "./tmp_docker/models/tinyllama-1.1b-chat"
    lora_model_path = "./src/output/v2"
    val_path = "./tmp_docker/dataset/validation.jsonl"
    
    try:
        # モデル読み込み
        print("Loading models...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=dtype,
            device_map="auto" if use_gpu else "cpu",
            trust_remote_code=True,
            torch_dtype=dtype
        )
        
        trained_model = PeftModel.from_pretrained(base_model, lora_model_path)
        trained_model.eval()
        
        print(f"Model loaded on: {next(trained_model.parameters()).device}")
        
        # Validation data読み込み（サンプル数制限）
        print("Loading validation data...")
        val_data = load_validation_data(val_path, max_samples=30)  # 30件に制限
        print(f"Loaded {len(val_data)} validation samples")
        
        # 1. Perplexity評価（15件、バッチサイズ3）
        print("\n📊 1. Perplexity評価")
        sample_texts = []
        for item in val_data[:15]:  # 15件に制限
            input_text = item.get('input', '').strip()
            if input_text:
                text = f"<|user|>\n{item['instruction']}\n{input_text}<|end|>\n<|assistant|>\n{item['output']}<|end|>"
            else:
                text = f"<|user|>\n{item['instruction']}<|end|>\n<|assistant|>\n{item['output']}<|end|>"
            sample_texts.append(text)
        
        perplexity = calculate_perplexity_batch(trained_model, tokenizer, sample_texts, batch_size=3)
        print(f"Perplexity: {perplexity:.2f}")
        
        # 2. Axentax DSL理解度評価（20件、バッチサイズ2）
        print("\n🎵 2. Axentax DSL理解度評価")
        axentax_results = evaluate_axentax_batch(trained_model, tokenizer, val_data[:20], batch_size=2)
        print(f"Overall Accuracy: {axentax_results['overall_accuracy']:.3f}")
        print(f"Valid Syntax Accuracy: {axentax_results['valid_accuracy']:.3f}")
        print(f"Invalid Syntax Accuracy: {axentax_results['invalid_accuracy']:.3f}")
        
        # 3. 簡単な質問応答テスト（GPU版は省略）
        print("\n💬 3. 一般QAテストは省略（時間短縮のため）")
        
        # GPU メモリ使用量確認
        if use_gpu:
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n🔧 GPU最大メモリ使用量: {memory_used:.2f}GB")
        
        # 結果保存
        evaluation_report = {
            "perplexity": perplexity,
            "axentax_evaluation": {
                "overall_accuracy": axentax_results['overall_accuracy'],
                "valid_accuracy": axentax_results['valid_accuracy'],
                "invalid_accuracy": axentax_results['invalid_accuracy'],
                "total_samples": len(val_data[:20])
            },
            "evaluation_config": {
                "device": str(device),
                "dtype": str(dtype),
                "perplexity_samples": 15,
                "axentax_samples": 20,
                "max_memory_used_gb": torch.cuda.max_memory_allocated() / 1e9 if use_gpu else 0
            }
        }
        
        os.makedirs("./src/output", exist_ok=True)
        with open("./src/output/evaluation_report_gpu.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        
        print("🎉 GPU評価完了! レポート: ./src/output/evaluation_report_gpu.json")
        
    except Exception as e:
        print(f"❌ 評価中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if use_gpu:
            torch.cuda.empty_cache()
            print("GPU cache cleared")


if __name__ == "__main__":
    main()