#!/usr/bin/env python3
"""
TinyLlama 1.1B 学習済みLoRAモデル 推論テスト（修正版）

修正内容:
- pad_token設定追加
- attention_mask対応
- 推論時のLoRA状態確認
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")


def main():
    print("🎯 TinyLlama 1.1B 学習済みLoRAモデル 推論テスト（修正版）")
    print("=" * 60)
    
    base_model_path = "./tmp_docker/models/tinyllama-1.1b-chat"
    lora_model_path = "./src/output/v2"
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"pad_token: {tokenizer.pad_token}")
        print(f"eos_token: {tokenizer.eos_token}")
        
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        
        # LoRA状態確認
        print(f"LoRA merged: {model.peft_config}")
        print(f"Active adapters: {model.active_adapters}")
        
        model.eval()
        
        print("\n📝 学習済みモデル推論テスト")
        
        # シンプルなテストケース
        test_cases = [
            "<|user|>\nこんにちは。あなたは何ができますか？<|end|>\n<|assistant|>\n",
            "<|user|>\n簡単な質問です。1+1は？<|end|>\n<|assistant|>\n",
            "<|user|>\n音楽について何か教えて<|end|>\n<|assistant|>\n"
        ]
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\n--- テスト {i} ---")
            print(f"入力: {prompt}")
            
            # トークナイズ
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128,
                padding=True
            )
            input_length = inputs["input_ids"].shape[1]
            
            print(f"入力長: {input_length} トークン")
            print(f"attention_mask shape: {inputs['attention_mask'].shape}")
            
            # 生成
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=50,
                        do_sample=False,  # グリーディ生成で安定化
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # 新しく生成された部分のみデコード
                    generated_tokens = outputs[0][input_length:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    print(f"生成結果: '{response}'")
                    
                    if response.strip():
                        print("✅ 正常生成")
                    else:
                        print("⚠️  空の応答")
                        
                except Exception as e:
                    print(f"❌ 生成エラー: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\n📊 モデル情報:")
        print(f"ベースモデル: {base_model_path}")
        print(f"LoRAアダプタ: {lora_model_path}")
        
        # パラメータ確認（正しい方法）
        total_params = sum(p.numel() for p in base_model.parameters())
        
        # LoRA adapter parameters
        lora_params = 0
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                lora_params += param.numel()
        
        print(f"ベースモデルパラメータ: {total_params:,}")
        print(f"LoRAパラメータ: {lora_params:,}")
        
        print("\n🎉 修正版推論テスト完了！")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()