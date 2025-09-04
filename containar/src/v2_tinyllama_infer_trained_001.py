#!/usr/bin/env python3
"""
TinyLlama 1.1B 学習済みLoRAモデル 推論テスト

実行方法:
  python src/v2_tinyllama_infer_trained_001.py

学習済みモデル:
  - ベースモデル: ./tmp_docker/models/tinyllama-1.1b-chat
  - LoRAアダプタ: ./src/output/v2
  - CPU推論（設定通り）
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")


def main():
    print("🎯 TinyLlama 1.1B 学習済みLoRAモデル 推論テスト")
    print("=" * 60)
    
    base_model_path = "./tmp_docker/models/tinyllama-1.1b-chat"
    lora_model_path = "./src/output/v2"
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.float32,  # CPU推論
            device_map="cpu",
            trust_remote_code=True
        )
        
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model.eval()
        
        print("\n📝 学習済みモデル推論テスト")
        
        # テストケース（学習データに含まれていそうな形式）
        test_cases = [
            "<|user|>\nこんにちは<|end|>\n<|assistant|>\n",
            "<|user|>\n音楽理論について教えて<|end|>\n<|assistant|>\n",
            "<|user|>\n@@ 120 1/4 { C F G } この記法について説明して<|end|>\n<|assistant|>\n",
            "<|user|>\nコード進行を教えて<|end|>\n<|assistant|>\n"
        ]
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\n--- テスト {i} ---")
            print(f"プロンプト: {repr(prompt)}")
            
            # トークナイズ
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=True)
            input_length = inputs["input_ids"].shape[1]
            
            print(f"入力長: {input_length} トークン")
            
            # 生成
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # デコード
                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                    
                    print(f"生成結果: {repr(response)}")
                    print(f"全体: {repr(full_response)}")
                    
                    if response.strip():
                        print("✅ 正常生成")
                    else:
                        print("⚠️  空の応答")
                        
                except Exception as e:
                    print(f"❌ 生成エラー: {e}")
        
        print(f"\n📊 モデル情報:")
        print(f"ベースモデル: {base_model_path}")
        print(f"LoRAアダプタ: {lora_model_path}")
        
        # パラメータ確認
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"総パラメータ: {total_params:,}")
        print(f"訓練可能パラメータ: {trainable_params:,}")
        
        print("\n🎉 学習済みモデル推論テスト完了！")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()