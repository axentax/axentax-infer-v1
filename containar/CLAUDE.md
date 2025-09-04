## 目的
データセットを使用してGemma-3 270Mをファインチューニングする。まずは1エポック
ディスク容量が無駄になるので学習の途中ステップは保存しなくて良い。

## ファインチューニング詳細
LoRAを使用する

## ソースにバージョン接頭辞を付与する。改変した場合は設備じの番号をインクリメント
v1_lora_finetune_000.py
v1_lora_finetune_001_simple.py
v1_infer_000.py
など

## 置き場所
- チューニング済みのモデルの置き場所\
./src/output

- ソースコードの置き場所
./src

- 元モデル\
tmp_docker/models

- データセット
tmp_docker/dataset

## 節約
学習はGPUを使用
推論はGPUを使用しない

## メモリエラー対処方法
CLAUDE_memory_error.md

## todo
TinyLlama 1.1B
