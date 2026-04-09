# -*- coding: utf-8 -*-
"""
llm_scorer.py — LLM 三元组真实性评分器（替代KGC教师模型）

功能：
    加载微调后的 Qwen3-4B LoRA 模型，对KG三元组进行批量真实性评分。
    输出 0~1 连续分数，通过 MMD 损失蒸馏到 Potential KG 的 APL。

使用时机：
    每3个epoch，Potential KG更新后，对所有候选三元组预计算评分并缓存。
    训练时只需查表，无需在线推理。
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.relation_templates import REL_TEMPLATES
from utils.memory_monitor import log_cuda_mem


SYSTEM_PROMPT = (
    "You are a knowledge graph plausibility scoring expert. "
    "Given the following statement about a book or literary work, "
    "rate the plausibility of this relationship on a scale from 0 to 1, "
    "where 0 means completely implausible and 1 means fully plausible.\n"
    "Answer with only a single number: 0 or 1."
)

RDF_TYPE_ALIAS = '22-rdf-syntax-ns#type'
RDF_TYPE_TARGET = 'type.object.type'


class LLMScorer:
    """
    LLM 三元组评分器（替代KGC教师模型）
    
    __init__ 时加载模型（一次性），之后通过 score_triplets() 对三元组批量评分。
    
    参数:
        model_path: 基座模型路径
        adapter_path: LoRA adapter 路径
        data_dir: 数据目录（如 data/amazon-book/），用于加载实体名和关系名
        device: GPU设备
        batch_size: 批量推理大小
    """
    
    def __init__(self, model_path, adapter_path, data_dir, device="cuda:0", batch_size=256, mem_debug=False):
        self.device = device
        self.batch_size = batch_size
        self.mem_debug = mem_debug
        
        # ===== 加载实体名和关系名（用于三元组→文本转换）=====
        self.entity_names, self.rel_names = self._load_names(data_dir)
        print(f"[LLM-Scorer] 实体名: {len(self.entity_names)}, 关系名: {len(self.rel_names)}")
        
        # ===== 加载模型 =====
        print(f"[LLM-Scorer] 加载基座: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.zero_token_id = self.tokenizer.encode("0", add_special_tokens=False)[0]
        self.one_token_id = self.tokenizer.encode("1", add_special_tokens=False)[0]
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        )
        
        if adapter_path and os.path.exists(adapter_path):
            print(f"[LLM-Scorer] 加载 LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            print("[LLM-Scorer] LoRA 已合并")
        
        model.eval()
        self.model = model
        print("[LLM-Scorer] 模型就绪")
        log_cuda_mem("llm_scorer.after_model_ready", self.device, self.mem_debug)
    
    def _load_names(self, data_dir):
        """加载实体名和关系名"""
        rel_names = {}
        with open(os.path.join(data_dir, 'relation_list.txt')) as f:
            next(f)
            for line in f:
                parts = line.strip().rsplit(maxsplit=1)
                if len(parts) == 2:
                    org, rid = parts[0], int(parts[1])
                    short = org.split('/')[-1] if '/' in org else org
                    if short == RDF_TYPE_ALIAS:
                        short = RDF_TYPE_TARGET
                    rel_names[rid] = short
        
        from utils.llm_data_utils import load_item_meta, load_kg, build_entity_names
        item_ids, isbn_map, isbn_to_meta = load_item_meta(data_dir)
        triplets = load_kg(data_dir)
        entity_names = build_entity_names(item_ids, isbn_map, isbn_to_meta, triplets, rel_names, data_dir)
        
        return entity_names, rel_names
    
    def _triplet_to_text(self, h, r, t):
        """将单个三元组 (h, r, t) 转为自然语言文本"""
        rn = self.rel_names.get(r)
        if rn is None or rn not in REL_TEMPLATES:
            return None
        tpl = REL_TEMPLATES[rn]
        if not tpl:
            return None
        h_name = self.entity_names.get(h)
        t_name = self.entity_names.get(t)
        if not h_name or not t_name:
            return None
        return tpl.format(head=h_name, tail=t_name)
    
    def _build_prompt(self, text):
        """构造与训练格式一致的 chat prompt"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    
    @torch.no_grad()
    def score_triplets(self, triplets_tensor, target_device=None):
        """
        对一批三元组进行 LLM 真实性评分
        
        参数:
            triplets_tensor: [N, 3] LongTensor，每行 [head, relation, tail]
            target_device: 返回结果放在哪个device上
        
        返回:
            scores: [N, 1] FloatTensor，每条三元组的真实性分数 ∈ [0, 1]
                   无法翻译的三元组默认 0.5（不确定）
        """
        if target_device is None:
            target_device = triplets_tensor.device
        
        triplets_np = triplets_tensor.cpu().numpy()
        n = len(triplets_np)
        scores = np.full(n, 0.5, dtype=np.float32)  # 默认0.5
        
        # 预处理：对所有三元组生成文本
        texts = []
        valid_indices = []
        for i in range(n):
            h, r, t = int(triplets_np[i, 0]), int(triplets_np[i, 1]), int(triplets_np[i, 2])
            text = self._triplet_to_text(h, r, t)
            if text is not None:
                texts.append(text)
                valid_indices.append(i)
        
        if not texts:
            print(f"[LLM-Scorer] 警告: {n}条三元组均无法翻译为文本")
            return torch.FloatTensor(scores).unsqueeze(-1).to(target_device)
        
        # 批量推理
        prompts = [self._build_prompt(t) for t in texts]
        
        for batch_start in tqdm(range(0, len(prompts), self.batch_size), 
                                desc="LLM评分", leave=False):
            batch_prompts = prompts[batch_start:batch_start + self.batch_size]
            batch_valid_idx = valid_indices[batch_start:batch_start + self.batch_size]
            batch_id = batch_start // self.batch_size
            
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(self.device)
            if self.mem_debug and (batch_id == 0 or (batch_id + 1) % 500 == 0):
                log_cuda_mem(f"llm_scorer.batch{batch_id}.after_tokenize", self.device, True,
                             extra=f"batch_size={len(batch_prompts)}")
            
            outputs = self.model(**inputs, return_dict=True)
            if self.mem_debug and (batch_id == 0 or (batch_id + 1) % 500 == 0):
                log_cuda_mem(f"llm_scorer.batch{batch_id}.after_forward", self.device, True)
            next_token_logits = outputs.logits[:, -1, :]
            binary_logits = next_token_logits[:, [self.zero_token_id, self.one_token_id]]
            binary_probs = torch.softmax(binary_logits.float(), dim=-1)
            one_probs = binary_probs[:, 1].detach().cpu().numpy()

            for j, prob_one in enumerate(one_probs):
                scores[batch_valid_idx[j]] = float(prob_one)
        
        n_scored = len(valid_indices)
        mean_score = scores[valid_indices].mean() if valid_indices else 0
        print(f"[LLM-Scorer] 评分完成: {n}条, "
              f"可翻译={n_scored}, 均值={mean_score:.3f}, "
              f"不可翻译={n - n_scored}(默认0.5)")
        
        return torch.FloatTensor(scores).unsqueeze(-1).to(target_device)
    
    def unload(self):
        """释放模型显存"""
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
            print("[LLM-Scorer] 模型已卸载，显存已释放")
