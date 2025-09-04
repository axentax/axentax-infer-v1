# TinyLlama ファインチューニング プロジェクト

## 概要
TinyLlama 1.1Bモデルを使用したAxentax DSLデータセットでのファインチューニング実装。

## プロジェクト構成

### スクリプト
- `v2_000_tinyllama_finetune_validation.py` - Validation監視付き学習
- `v2_002_tinyllama_finetune_5epoch.py` - 5エポック学習（推奨）
- `v2_000_tinyllama_infer.py` - 推論テスト
- `v2_001_evaluation_gpu.py` - GPU使用評価
- `v2_000_evaluation.py` - 総合評価

### データ配置
```
tmp_docker/
├── models/
│   └── tinyllama-1.1b-chat/     # ベースモデル
└── dataset/
    ├── train.jsonl              # 学習データ (13,525例)
    ├── axentax_full.jsonl       # 追加学習データ (15,028例)
    └── validation.jsonl         # 検証データ (1,503例)

src/
└── output/                      # 学習済みモデル出力先
```

## 学習進行状況の監視方法

### 1. リアルタイム進捗確認
```bash
# 学習の詳細出力を確認（BashOutputツール使用）
# 現在のステップ数、loss値、学習速度が表示される

# プロセス確認
ps aux | grep tinyllama

# GPU使用状況
nvidia-smi
```

### 2. ステップ進捗の目安
```bash
# 5エポック学習の場合:
# 総ステップ数: 17,850
# 現在進捗の確認: 学習ログの最後の行を参照
# 例: "173/17850 [xx:xx<xx:xx:xx, x.xxs/it]" 

# 進捗率計算例:
# (現在ステップ / 17850) × 100 = 進捗パーセンテージ
```

### 3. 学習状況の判断指標
```bash
# 正常な学習の兆候:
# - GPU使用率: 80-100%
# - メモリ使用: 3-6GB程度（RTX 5070の場合）
# - CPU使用率: 90%以上
# - 学習速度: 1.0-1.1s/it

# 異常の兆候:
# - プロセスが停止
# - GPU使用率0%
# - エラーメッセージの出力
```

### 4. 出力ファイルの確認
```bash
# 学習済みモデルの保存確認
ls -la src/output/v2_5epoch/

# 中間保存ファイル（1000ステップごと）
ls -la src/output/v2_5epoch/checkpoint-*/

# ログファイル
ls -la src/output/v2_5epoch/logs/
```

### 5. validation loss の監視
```bash
# 初期validation loss: 通常2.5-3.0程度
# 学習進行に伴い徐々に減少
# 過学習の兆候: validation lossが上昇に転じる
```

## 学習完了後の確認

### 1. 推論テスト
```bash
python src/v2_000_tinyllama_infer.py
```

### 2. 総合評価
```bash
# GPU版評価（推奨）
python src/v2_001_evaluation_gpu.py

# CPU版評価
python src/v2_000_evaluation.py
```

### 3. 評価結果の確認
```bash
cat src/output/evaluation_report_gpu.json
```

## トラブルシューティング

### メモリ不足の場合
1. `CLAUDE_memory_error.md`を参照
2. バッチサイズを削減
3. gradient_accumulation_stepsを調整

### 学習が停止した場合
```bash
# プロセス確認
ps aux | grep python

# ログ確認
tail -100 <学習ログ>

# GPU状況確認
nvidia-smi

# 再実行
python src/v2_002_tinyllama_finetune_5epoch.py
```

### 学習時間の短縮
```bash
# より少ないエポックで学習
# スクリプト内の num_epochs を調整

# より小さなデータセットでテスト
# データパスを変更してサブセットを使用
```

## ハイパーパラメータ調整指針

### LoRA設定
- r=16: 標準的な設定
- r=32: より高い表現力（メモリ使用量増加）
- learning_rate: 5e-5〜2e-4

### 学習設定  
- max_seq_length: 512-1024（長文データが多い場合は2048）
- エポック数: 1-5（validation lossで早期停止）
- バッチサイズ: 1-2（GPU メモリに応じて調整）

## 注意事項
- 学習には約4-6時間かかる見込み
- GPU使用を推奨（CPU学習は非常に時間がかかる）
- 定期的に進捗を確認し、異常があれば早期に対処する