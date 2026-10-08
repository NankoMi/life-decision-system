# -*- coding: utf-8 -*-
"""
Laya 决策模型 CLI 桥接（life-decision-system 第3层）
用法：
  python laya_bridge.py choice --options "选项A,选项B,选项C" --question "该选哪个" --context "背景描述"
  python laya_bridge.py score  --question "这件事值得做吗" --context "背景描述" [--scale "1=强烈反对,3=中立,5=强烈支持"]
  python laya_bridge.py noul   --question "这样会不会违法" --context "背景描述"
输出：JSON（stdout），便于上层解析。
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCALE = ["1=强烈反对", "2=不太赞成", "3=中立", "4=比较赞成", "5=强烈支持"]


def resolve_model(explicit: str = None) -> str:
    """模型路径解析：--base > 环境变量 LAYA_MODEL > skill内微调版 > skill内底模 > HF在线"""
    if explicit:
        return explicit
    env = os.environ.get("LAYA_MODEL")
    if env and os.path.exists(env):
        return env
    ft = sorted([d for d in os.listdir(os.path.join(_HERE, "finetune"))
                 if d.startswith("ckpt-")]) if os.path.isdir(os.path.join(_HERE, "finetune")) else []
    if ft:
        return os.path.join(_HERE, "finetune", ft[-1])  # 最新微调版
    local = os.path.join(_HERE, "laya-model")
    if os.path.isdir(local) and os.path.exists(os.path.join(local, "model.safetensors")):
        return local
    return "convaiinnovations/laya"  # HF 在线（首次自动下载）

# score 缺省量表：1~5
DEFAULT_SCALE = ["1=强烈反对", "2=不太赞成", "3=中立", "4=比较赞成", "5=强烈支持"]


def build_question(qtype: str, question: str, options=None, scale=None) -> dict:
    if qtype == "choice":
        criteria = {opt.split("=", 1)[0].strip(): opt.split("=", 1)[1].strip() if "=" in opt else opt.strip()
                    for opt in options}
        return {"type": "choice", "instructions": question, "criteria": criteria}
    if qtype == "score":
        return {"type": "score", "instructions": question, "criteria": scale or DEFAULT_SCALE}
    if qtype == "noul":
        return {"type": "noul", "instructions": question}
    raise ValueError(f"unknown qtype: {qtype}")


def main() -> int:
    p = argparse.ArgumentParser(description="Laya decision model CLI")
    p.add_argument("qtype", choices=["choice", "score", "noul"])
    p.add_argument("--question", required=True, help="判断问题")
    p.add_argument("--context", default="", help="状态/背景描述")
    p.add_argument("--options", help="choice 用：逗号分隔，可写 'A=描述A,B=描述B'")
    p.add_argument("--scale", help="score 用：逗号分隔，如 '1=反对,5=支持'")
    p.add_argument("--base", help="覆盖模型路径（默认自动解析：环境变量LAYA_MODEL > 微调版 > 底模 > HF在线）")
    args = p.parse_args()

    if args.qtype == "choice" and not args.options:
        print(json.dumps({"error": "choice 需要 --options"}, ensure_ascii=False), file=sys.stderr)
        return 2

    import laya  # 延迟导入，加快 --help

    agent = laya.load(resolve_model(getattr(args, "base", None)))
    qdef = build_question(args.qtype, args.question,
                          options=(args.options.split(",") if args.options else None),
                          scale=(args.scale.split(",") if args.scale else None))
    result = agent.predict(args.context or "(无背景)", {"decision": qdef})
    a = result["answers"]["decision"]  # {'type','choice'/'score'/'noul','probabilities','confidence',...}

    answer_key = {"choice": "choice", "score": "score", "noul": "noul"}[args.qtype]
    out = {
        "qtype": args.qtype,
        "question": args.question,
        "answer": a.get(answer_key),
        "confidence": a.get("confidence"),
        "probabilities": a.get("probabilities"),
        "raw": a,
    }
    print(json.dumps(out, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
