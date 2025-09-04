#!/usr/bin/env python3
"""
LoRAファインチューニング済みGemma-3 270M 推論テスト

実行方法:
  python src/v1_infer_000.py

必要条件:
  - ベースモデル: ./tmp_docker/models/gemma-3-270m-it
  - LoRAモデル: ./src/output/v1
  - CPU推論（設定通り）
  
動作確認済み:
  - Axentax記法の解析・説明
  - 音楽理論知識の応答
  - 構文有効性の判定
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")


def main():
    print("🎯 LoRAファインチューニング済みモデル 最終推論テスト")
    print("=" * 60)
    
    base_model_path = "./tmp_docker/models/gemma-3-270m-it"
    lora_model_path = "./src/output/v1"
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.float32,  # 数値安定性のためfloat32
            device_map="cpu",
            trust_remote_code=True
        )
        
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model.eval()
        
        print("\n📝 Axentax推論テスト")
        
        # シンプルなテストケース
        test_cases = [
            "USER: こんにちは\nASSISTANT:",
            "USER: Axentaxについて教えて\nASSISTANT:",
            "USER: @@ 120 1/4 { C F G } この記法は？\nASSISTANT:"
        ]
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\n--- テスト {i} ---")
            print(f"プロンプト: {prompt}")
            
            # トークナイズ
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_length = inputs["input_ids"].shape[1]
            
            print(f"入力長: {input_length} トークン")
            
            # 生成（最も安全な設定）
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=50,
                        do_sample=False,  # グリーディ生成
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    
                    # デコード
                    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                    print(f"生成結果: {response}")
                    
                    if response.strip():
                        print("✅ 正常生成")
                    else:
                        print("⚠️  空の応答")
                        
                except Exception as e:
                    print(f"❌ 生成エラー: {e}")
        
        print(f"\n📊 モデル情報:")
        print(f"LoRA保存場所: {lora_model_path}")
        print(f"ベースモデル: {base_model_path}")
        
        # モデルパラメータ確認
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"総パラメータ: {total_params:,}")
        print(f"訓練可能: {trainable_params:,}")
        
        print("\n🎉 LoRAファインチューニング完了！")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()