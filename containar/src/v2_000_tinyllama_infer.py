#!/usr/bin/env python3
"""
TinyLlama 1.1B 推論テスト

実行方法:
  python src/v2_tinyllama_infer_000.py

必要条件:
  - ベースモデル: ./tmp_docker/models/tinyllama-1.1b-chat
  - CPU推論（設定通り）
  
動作確認:
  - TinyLlama チャット形式での会話
  - Axentax記法の理解テスト
  - 音楽理論知識の応答
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")


def main():
    print("🎯 TinyLlama 1.1B ベースモデル 推論テスト")
    print("=" * 60)
    
    model_path = "./tmp_docker/models/tinyllama-1.1b-chat"
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,  # CPU推論
            device_map="cpu",
            trust_remote_code=True
        )
        model.eval()
        
        print("\\n📝 TinyLlama推論テスト")
        
        # TinyLlama chat format test cases
        test_cases = [
            "<|user|>\\nこんにちは<|end|>\\n<|assistant|>\\n",
            "<|user|>\\nAxentaxについて教えて<|end|>\\n<|assistant|>\\n",
            "<|user|>\\n@@ 120 1/4 { C F G } この記法は何ですか？<|end|>\\n<|assistant|>\\n"
        ]
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\\n--- テスト {i} ---")
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
                        max_new_tokens=80,
                        do_sample=False,  # グリーディ生成
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                        temperature=1.0,
                        top_p=0.9
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
        
        print(f"\\n📊 モデル情報:")
        print(f"モデルパス: {model_path}")
        
        # モデルパラメータ確認
        total_params = sum(p.numel() for p in model.parameters())
        print(f"総パラメータ: {total_params:,}")
        
        print("\\n🎉 TinyLlama推論テスト完了！")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()