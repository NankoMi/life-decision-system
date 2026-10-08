# Laya 微调流程（life-decision-system）

## 数据源两条腿

1. **书数据集**（已建好，787 条）：`build_book_dataset.py` 把《高性价比人生指南》667 条条目转成
   score/noul 训练样本 → `book_dataset.jsonl`。标签是启发式的（收益量级+成本修正），够 v1 用。
2. **真实决策记录**（边用边攒）：每次我用 Laya 给你打分，你最终选了什么、结果如何，
   追加到 `decisions.jsonl`。攒到 200+ 条后并入训练集做 v2。

## 重新生成书数据集（书更新后）

```bash
cd <skill目录>/finetune
python build_book_dataset.py
```

## 训练命令（4070 TiS，787条×3轮 实测约几分钟）

```bash
cd <skill目录>/finetune
set HF_HUB_OFFLINE=1
python -m laya.train_cli ^
  --data book_dataset.jsonl --base ../laya-model --out ./ckpt-book-v1 ^
  --epochs 3 --device cuda
```

- 必须用 `python -m laya.train_cli`（直接跑脚本会 relative import 报错）
- 必须设 `HF_HUB_OFFLINE=1`（否则代理 502）
- 产出 `ckpt-book-v1/`，改 `../laya_bridge.py` 的 `MODEL_ID` 指过去即生效

## 真实决策记录格式（decisions.jsonl，每行一条）

```json
{"date": "2026-10-07", "question": "该选哪个offer", "qtype": "choice",
 "options": {"A": "...", "B": "..."}, "laya_answer": "A", "user_choice": "B",
 "outcome": "", "book_ref": "第4节第12条"}
```

`outcome` 事后补（比如三个月后回头填结果）。这就是"判断打分模型越用越准"的闭环。

## 微调注意事项（来自 9/23 调研结论）

- base 模型零样本接近随机（typed-decisions ~0.35），微调后 0.766 —— **不微调不能用**
- 微调后必须做温度校准：`laya.calibrate.fit_temperatures`，否则 ECE 0.466 → 校准后 0.081
- 选项 >20 个性能崩，出题时选项控制在个位数
- 英语 checkpoint 只认 512 token，context 别写太长
