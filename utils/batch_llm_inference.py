# -*- coding: utf-8 -*-
"""
batch_llm_inference.py — 用微调后的 Qwen3-4B LoRA 对 KG 三元组批量推理
"""
import os
import json
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from path_utils import default_adapter_path, default_base_model_path, default_dataset_root, resolve_dataset_dir


def load_model(model_path, adapter_path, device="cuda:0"):
    print(f"[模型] 加载基座: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )

    if adapter_path:
        print(f"[模型] 加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("[模型] LoRA 已合并")

    model.eval()
    return model, tokenizer


def build_prompt(messages, tokenizer):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def main():
    parser = argparse.ArgumentParser()
    default_data_dir = resolve_dataset_dir(default_dataset_root(), 'amazon-book')
    parser.add_argument('--model_path', default=default_base_model_path())
    parser.add_argument('--adapter_path', default=default_adapter_path())
    parser.add_argument('--input_path', default=os.path.join(default_data_dir, 'llm_data', 'inference_swift.jsonl'))
    parser.add_argument('--output_path', default=os.path.join(default_data_dir, 'llm_data', 'inference_predictions.jsonl'))
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='从上次中断处继续')
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    model, tokenizer = load_model(args.model_path, args.adapter_path, device)

    # 加载数据
    print(f"[数据] 读取: {args.input_path}")
    data = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"[数据] 总共: {len(data)} 条")

    # 断点续推
    done_ids = set()
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    done_ids.add(item['custom_id'])
                except:
                    pass
        print(f"[续推] 已完成: {len(done_ids)} 条")
        data = [d for d in data if d['custom_id'] not in done_ids]
        print(f"[续推] 剩余: {len(data)} 条")

    if not data:
        print("[完成] 无需推理")
        return

    # 流式写入 + 批量推理
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    mode = 'a' if args.resume and done_ids else 'w'
    
    true_count = 0
    false_count = 0
    other_count = 0
    total = 0
    t0 = time.time()

    with open(args.output_path, mode, encoding='utf-8') as fout:
        n_batches = (len(data) + args.batch_size - 1) // args.batch_size
        for i in tqdm(range(0, len(data), args.batch_size), desc="批量推理", total=n_batches):
            batch = data[i:i + args.batch_size]
            prompts = [build_prompt(item['messages'], tokenizer) for item in batch]
            ids = [item['custom_id'] for item in batch]

            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=256,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    temperature=None, top_p=None, pad_token_id=tokenizer.pad_token_id,
                )

            input_len = inputs['input_ids'].shape[1]
            for j, output in enumerate(outputs):
                text = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                fout.write(json.dumps({"custom_id": ids[j], "prediction": text}, ensure_ascii=False) + '\n')
                
                pl = text.lower()
                if pl in ('true', '1', 'yes'):
                    true_count += 1
                elif pl in ('false', '0', 'no'):
                    false_count += 1
                else:
                    other_count += 1
                total += 1

            # 每 1000 batch 刷新 + 打印进度
            if (i // args.batch_size) % 1000 == 0 and i > 0:
                fout.flush()
                elapsed = time.time() - t0
                speed = total / elapsed
                eta = (len(data) - total) / speed if speed > 0 else 0
                print(f"\n[进度] {total}/{len(data)} | True={true_count} False={false_count} Other={other_count} | "
                      f"速度={speed:.0f}条/s | ETA={eta/3600:.1f}h")

    elapsed = time.time() - t0
    print(f"\n[完成] 总计: {total} 条, 耗时: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  True:  {true_count} ({true_count/total*100:.1f}%)")
    print(f"  False: {false_count} ({false_count/total*100:.1f}%)")
    if other_count > 0:
        print(f"  Other: {other_count} ({other_count/total*100:.1f}%)")


if __name__ == '__main__':
    main()
