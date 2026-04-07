# -*- coding: utf-8 -*-
"""
generate_hard_negatives.py — 用LLM生成对抗性困难负样本
针对不同关系类型设计不同的prompt，让LLM生成语义相似但错误的替换实体。
"""
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from modelscope import AutoModelForCausalLM, AutoTokenizer

# ======== 关系分组 ========
# 每组关系有对应的LLM prompt策略
RELATION_GROUPS = {
    'author': {
        'book.written_work.author',      # book -> author
        'book.author.works_written',      # author -> book
    },
    'genre': {
        'book.book.genre',
        'media_common.literary_genre.books_in_this_genre',
        'book.short_story.genre',
        'media_common.literary_genre.stories_in_this_genre',
        'theater.play.genre',
        'theater.theater_genre.plays_in_this_genre',
    },
    'subject': {
        'book.written_work.subjects',
        'book.book_subject.works',
    },
    'series': {
        'book.written_work.part_of_series',
        'book.literary_series.works_in_this_series',
        'book.written_work.previous_in_series',
        'book.written_work.next_in_series',
    },
    'character': {
        'book.book.characters',
        'book.book_character.appears_in_book',
    },
    'language': {
        'book.written_work.original_language',
    },
    'other': {
        'book.book.interior_illustrations_by',
        'book.illustrator.books_illustrated',
        'fictional_universe.work_of_fiction.part_of_these_fictional_universes',
        'fictional_universe.fictional_universe.works_set_here',
        'theater.play.country_of_origin',
    },
}

# 每种关系组的prompt模板：给定实体，让LLM生成相似但不同的替换实体
HARD_NEG_PROMPTS = {
    'author': (
        'The author "{entity}" writes books. '
        'Name 5 other real authors who write in a similar genre or era, but are DIFFERENT people. '
        'Reply with ONLY the author names, one per line. No numbering, no explanation.'
    ),
    'genre': (
        '"{entity}" is a book genre/category. '
        'Name 5 other book genres or categories that are closely related but DIFFERENT. '
        'Reply with ONLY the genre names, one per line. No numbering, no explanation.'
    ),
    'subject': (
        '"{entity}" is a book subject/topic category. '
        'Name 5 other subject categories that are related but DIFFERENT. '
        'Reply with ONLY the category names, one per line. No numbering, no explanation.'
    ),
    'series': (
        '"{entity}" is a book series name. '
        'Name 5 other real book series that are in a similar genre but by DIFFERENT authors. '
        'Reply with ONLY the series names, one per line. No numbering, no explanation.'
    ),
    'character': (
        '"{entity}" is a fictional character from a book. '
        'Name 5 other fictional book characters from similar genres but DIFFERENT books. '
        'Reply with ONLY the character names, one per line. No numbering, no explanation.'
    ),
    'language': (
        '"{entity}" is a language. '
        'Name 5 other languages that are geographically or linguistically related but DIFFERENT. '
        'Reply with ONLY the language names, one per line. No numbering, no explanation.'
    ),
    'other': (
        '"{entity}" is related to books or literature. '
        'Name 5 similar but different entities of the same type. '
        'Reply with ONLY the names, one per line. No numbering, no explanation.'
    ),
}


def get_relation_group(rel_name):
    """根据关系名返回所属组别"""
    for group, rels in RELATION_GROUPS.items():
        if rel_name in rels:
            return group
    return None


def load_model(model_path='/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B'):
    print(f"[LLM] Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def generate_similar_entities(model, tokenizer, entity_name, group, num=5):
    """让LLM生成与给定实体语义相似的替换实体"""
    prompt_template = HARD_NEG_PROMPTS.get(group, HARD_NEG_PROMPTS['other'])
    prompt = prompt_template.format(entity=entity_name)

    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=150, do_sample=True,
            temperature=0.7, top_p=0.9, repetition_penalty=1.1
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # 解析：每行一个实体名
    entities = []
    for line in response.strip().split('\n'):
        line = line.strip().strip('-').strip('•').strip('*').strip()
        # 去掉数字编号
        if line and line[0].isdigit():
            line = line.lstrip('0123456789.').strip()
        if line and len(line) > 1 and len(line) < 100:
            entities.append(line)
    return entities[:num]


def batch_generate_hard_negatives(data_dir, output_path, max_samples=2500, seed=2023):
    """
    为训练数据生成LLM对抗性困难负样本。

    流程：
    1. 加载KG数据和实体名称
    2. 收集需要生成困难负样本的(实体, 关系组)对
    3. 对每个独特实体调用一次LLM生成替换候选
    4. 用替换候选构造困难负样本语句
    """
    random.seed(seed)
    np.random.seed(seed)

    # 动态加载同目录下的函数
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llm_data_utils import (
        load_item_meta, load_relations, load_kg, build_entity_names,
        make_nl_statement, REL_TEMPLATES, SYSTEM_PROMPT
    )

    item_ids, isbn_map, isbn_to_meta = load_item_meta(data_dir)
    rel_names, skip_rids = load_relations(data_dir)
    triplets = load_kg(data_dir)
    entity_names = build_entity_names(item_ids, isbn_map, isbn_to_meta, triplets, rel_names)

    # 筛选可翻译的三元组
    named_triplets = [(h, r, t) for h, r, t in triplets
                      if r not in skip_rids
                      and h in entity_names and t in entity_names
                      and rel_names.get(r) in REL_TEMPLATES]
    triplet_set = set(triplets)
    print(f"可用三元组: {len(named_triplets)}")

    # 选取需要困难负样本的三元组
    selected = random.sample(named_triplets, min(max_samples, len(named_triplets)))

    # 收集需要LLM生成替换的独特(实体名, 关系组)对
    entity_group_pairs = {}  # (entity_name, group) -> set of entity_ids
    tasks = []  # [(h, r, t, replace_position, target_entity_id, target_name, group)]

    for h, r, t in selected:
        rn = rel_names.get(r)
        group = get_relation_group(rn)
        if group is None:
            continue

        # 决定替换哪个位置
        # author关系：替换author实体（非book实体）
        # genre/subject关系：替换genre/subject实体
        # 其他：随机选head或tail
        if group == 'author':
            if rn == 'book.written_work.author':
                # h=book, t=author -> 替换author(tail)
                target_id, target_name, pos = t, entity_names[t], 'tail'
            else:
                # h=author, t=book -> 替换author(head)
                target_id, target_name, pos = h, entity_names[h], 'head'
        elif group in ('genre', 'subject', 'language'):
            if rn in ('book.book.genre', 'book.short_story.genre', 'theater.play.genre',
                       'book.written_work.subjects', 'book.written_work.original_language'):
                target_id, target_name, pos = t, entity_names[t], 'tail'
            else:
                target_id, target_name, pos = h, entity_names[h], 'head'
        elif group == 'series':
            if rn == 'book.written_work.part_of_series':
                target_id, target_name, pos = t, entity_names[t], 'tail'
            elif rn == 'book.literary_series.works_in_this_series':
                target_id, target_name, pos = h, entity_names[h], 'head'
            else:
                # prev/next: 替换tail（另一本书）
                target_id, target_name, pos = t, entity_names[t], 'tail'
        elif group == 'character':
            if rn == 'book.book.characters':
                target_id, target_name, pos = t, entity_names[t], 'tail'
            else:
                target_id, target_name, pos = h, entity_names[h], 'head'
        else:
            pos = random.choice(['head', 'tail'])
            target_id = h if pos == 'head' else t
            target_name = entity_names[target_id]

        key = (target_name, group)
        if key not in entity_group_pairs:
            entity_group_pairs[key] = set()
        entity_group_pairs[key].add(target_id)
        tasks.append((h, r, t, pos, target_id, target_name, group))

    print(f"需要LLM生成替换的独特(实体,组)对: {len(entity_group_pairs)}")

    # 加载LLM
    model, tokenizer = load_model()

    # 批量生成替换候选
    replacement_cache = {}  # (entity_name, group) -> [replacement_name1, ...]
    for (ent_name, group) in tqdm(entity_group_pairs.keys(), desc="LLM生成替换实体"):
        replacements = generate_similar_entities(model, tokenizer, ent_name, group, num=5)
        replacement_cache[(ent_name, group)] = replacements

    # 释放显存
    del model
    torch.cuda.empty_cache()
    print("[LLM] Model unloaded, VRAM freed")

    # 构造困难负样本
    hard_negatives = []
    for h, r, t, pos, target_id, target_name, group in tasks:
        replacements = replacement_cache.get((target_name, group), [])
        if not replacements:
            continue

        replacement = random.choice(replacements)

        # 构造NL语句：把replacement直接填入模板
        rn = rel_names.get(r)
        if rn not in REL_TEMPLATES:
            continue

        h_name = entity_names.get(h)
        t_name = entity_names.get(t)
        if not h_name or not t_name:
            continue

        if pos == 'tail':
            stmt = REL_TEMPLATES[rn].format(head=h_name, tail=replacement)
        else:
            stmt = REL_TEMPLATES[rn].format(head=replacement, tail=t_name)

        hard_negatives.append({
            "role": "neg_hard",
            "input": stmt,
            "output": "False"
        })

    print(f"[困难负样本] 生成: {len(hard_negatives)}")

    # 写出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in hard_negatives:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"[困难负样本] 写出: {output_path}")
    return hard_negatives


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='amazon-book')
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--max_samples', type=int, default=2500,
                        help='困难负样本数量（总训练数据的~10%%）')
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()

    data_dir = os.path.join(args.data_path, args.dataset)
    output_path = os.path.join(data_dir, 'llm_data', 'hard_negatives.jsonl')
    batch_generate_hard_negatives(data_dir, output_path,
                                  max_samples=args.max_samples, seed=args.seed)
