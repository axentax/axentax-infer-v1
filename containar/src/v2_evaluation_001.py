#!/usr/bin/env python3
"""
TinyLlama 1.1B 学習済みモデル 評価スクリプト

評価項目:
1. Validation loss計算
2. BLEU/ROUGE等のメトリクス
3. Axentax DSL理解度テスト
4. 一般的な質問応答能力
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")
from datasets import Dataset
import numpy as np
from typing import List, Dict


def load_validation_data(val_path: str) -> List[Dict]:
    """Validation dataを読み込み"""
    data = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def calculate_perplexity(model, tokenizer, texts: List[str]) -> float:
    """Perplexityを計算"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            tokens = inputs["input_ids"].shape[1]
            
            total_loss += loss.item() * tokens
            total_tokens += tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()


def evaluate_axentax_understanding(model, tokenizer, test_cases: List[Dict]) -> Dict:
    """Axentax DSL理解度を評価"""
    model.eval()
    results = {
        "total": len(test_cases),
        "syntax_valid_correct": 0,
        "syntax_invalid_correct": 0,
        "detailed_results": []
    }
    
    for case in test_cases:
        instruction = case["instruction"]
        input_text = case.get("input", "")
        expected_output = case["output"]
        is_valid = case.get("meta", {}).get("valid", True)
        
        # プロンプト構築
        if input_text:
            prompt = f"<|user|>\n{instruction}\n{input_text}<|end|>\n<|assistant|>\n"
        else:
            prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"
        
        # 生成
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            try:
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=100,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
                
                # 評価ロジック
                if is_valid:
                    # 有効な構文の場合: "構文は有効です"が含まれるかチェック
                    if "構文は有効です" in response or "有効です" in response:
                        results["syntax_valid_correct"] += 1
                        correct = True
                    else:
                        correct = False
                else:
                    # 無効な構文の場合: "無効です"が含まれるかチェック
                    if "無効です" in response or "エラー" in response:
                        results["syntax_invalid_correct"] += 1
                        correct = True
                    else:
                        correct = False
                
                results["detailed_results"].append({
                    "input": input_text,
                    "expected_valid": is_valid,
                    "response": response,
                    "correct": correct
                })
                
            except Exception as e:
                results["detailed_results"].append({
                    "input": input_text,
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


def evaluate_general_qa(model, tokenizer, test_cases: List[str]) -> Dict:
    """一般的な質問応答能力を評価"""
    model.eval()
    results = {"responses": []}
    
    for question in test_cases:
        prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            try:
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
                
                results["responses"].append({
                    "question": question,
                    "response": response
                })
                
            except Exception as e:
                results["responses"].append({
                    "question": question,
                    "response": f"ERROR: {e}"
                })
    
    return results


def main():
    print("🔍 TinyLlama 1.1B 学習済みモデル 総合評価")
    print("=" * 60)
    
    # パス設定
    base_model_path = "./tmp_docker/models/tinyllama-1.1b-chat"
    lora_model_path = "./src/output/v2"  # 既存の学習済みモデル
    val_path = "./tmp_docker/dataset/validation.jsonl"
    
    try:
        # モデル読み込み
        print("Loading models...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        trained_model = PeftModel.from_pretrained(base_model, lora_model_path)
        trained_model.eval()
        
        # Validation data読み込み
        print("Loading validation data...")
        val_data = load_validation_data(val_path)
        print(f"Loaded {len(val_data)} validation samples")
        
        # 1. Perplexity評価（サンプル50件）
        print("\n📊 1. Perplexity評価")
        sample_texts = []
        for item in val_data[:50]:  # 計算量を考慮して50件
            input_text = item.get('input', '').strip()
            if input_text:
                text = f"<|user|>\n{item['instruction']}\n{input_text}<|end|>\n<|assistant|>\n{item['output']}<|end|>"
            else:
                text = f"<|user|>\n{item['instruction']}<|end|>\n<|assistant|>\n{item['output']}<|end|>"
            sample_texts.append(text)
        
        perplexity = calculate_perplexity(trained_model, tokenizer, sample_texts)
        print(f"Perplexity: {perplexity:.2f}")
        
        # 2. Axentax DSL理解度評価（サンプル100件）
        print("\n🎵 2. Axentax DSL理解度評価")
        axentax_results = evaluate_axentax_understanding(trained_model, tokenizer, val_data[:100])
        print(f"Overall Accuracy: {axentax_results['overall_accuracy']:.3f}")
        print(f"Valid Syntax Accuracy: {axentax_results['valid_accuracy']:.3f}")
        print(f"Invalid Syntax Accuracy: {axentax_results['invalid_accuracy']:.3f}")
        
        # 3. 一般的な質問応答能力テスト
        print("\n💬 3. 一般的な質問応答能力テスト")
        general_questions = [
            "こんにちは、調子はどうですか？",
            "1+1は何ですか？",
            "日本の首都はどこですか？",
            "音楽の基本的な要素を教えて",
            "プログラミングとは何ですか？"
        ]
        
        general_results = evaluate_general_qa(trained_model, tokenizer, general_questions)
        
        for item in general_results["responses"]:
            print(f"Q: {item['question']}")
            print(f"A: {item['response'][:100]}...")
            print()
        
        # 結果保存
        evaluation_report = {
            "perplexity": perplexity,
            "axentax_evaluation": {
                "overall_accuracy": axentax_results['overall_accuracy'],
                "valid_accuracy": axentax_results['valid_accuracy'],
                "invalid_accuracy": axentax_results['invalid_accuracy']
            },
            "general_qa_samples": general_results["responses"][:3]  # 最初の3件のみ保存
        }
        
        with open("./src/output/evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        
        print("🎉 評価完了! 詳細レポートを ./src/output/evaluation_report.json に保存しました")
        
    except Exception as e:
        print(f"❌ 評価中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()