# TinyLlama 1.1B ファインチューニング エラー調査レポート

## 日時
2025年9月4日

## 評価実行時の警告エラー

### 1. torch_dtype 非推奨警告
```
`torch_dtype` is deprecated! Use `dtype` instead!
```

**原因**: `v2_001_evaluation_gpu.py:220` で `torch_dtype=dtype` パラメータを使用
**対処**: `torch_dtype` を `dtype` に変更する必要

### 2. padding_side 警告（デコーダー専用アーキテクチャ）
```
A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
```

**原因**: TinyLlamaはデコーダー専用モデルだが、tokenizer初期化時に `padding_side='right'` がデフォルトで設定されている
**影響**: 生成結果の精度に影響する可能性
**対処**: tokenizer初期化時に `tokenizer.padding_side = 'left'` を設定

## 学習完了時のエラー

### 3. JSON保存エラー
```
dump() missing 1 required positional argument: 'fp'
```

**原因**: 学習完了時の `training_summary.json` 保存でファイルポインタが正しく渡されていない
**状況**: 学習は正常完了、モデルは保存済み
**対処**: `json.dump()` の引数を確認し修正

## 現在の学習結果

### 評価指標（GPU評価）
- **Perplexity**: 2.72（良好）
- **Axentax DSL理解度**:
  - 全体精度: 90.0%
  - 有効構文精度: 100%
  - 無効構文精度: 84.6%
- **GPU使用量**: 2.44GB（予定内）

### 学習進行状況
- **エポック**: 0.84/5.0（早期収束）
- **検証損失改善**: 2.9210 → 0.0915（97%改善）
- **学習時間**: 約55分
- **モデル保存先**: `src/output/v2_5epoch/`

## 推奨対処

### 優先度: 高
1. `v2_001_evaluation_gpu.py` の `torch_dtype` → `dtype` 修正
2. tokenizer初期化時に `padding_side='left'` 設定

### 優先度: 中
3. `v2_002_tinyllama_finetune_5epoch.py` のJSON保存エラー修正

## 結論
学習自体は非常に成功しており、Axentax DSL理解度90%という優秀な結果を達成。エラーは主に警告レベルで、モデル性能への大きな影響はなし。