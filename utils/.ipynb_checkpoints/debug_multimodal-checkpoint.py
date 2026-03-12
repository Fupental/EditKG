# -*- coding: utf-8 -*-
"""
调试脚本：验证多模态预处理流程
- ASIN 与 item_id 匹配是否正确
- BERT 文本编码器是否正常输出
- ViT 视觉编码器是否正常输出
只处理少量样本（默认5个），快速验证流程正确性。

使用方式：
    cd EditKG-main
    python utils/debug_multimodal.py
"""

import os
import sys
import json
import numpy as np
import torch

# 确保可以从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'amazon-book')
N_SAMPLES = 5  # 调试用样本数


def step1_check_asin_mapping():
    """Step 1: 检查 ASIN → item_id 映射"""
    print("=" * 60)
    print("Step 1: 检查 ASIN → item_id 映射")
    print("=" * 60)

    item_list_path = os.path.join(DATA_DIR, 'item_list.txt')
    if not os.path.exists(item_list_path):
        print(f"  ✗ 文件不存在: {item_list_path}")
        return None

    asin2id = {}
    with open(item_list_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        print(f"  表头: {header}")
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                parts = line.strip().split()
            if len(parts) >= 2:
                asin2id[parts[0]] = int(parts[1])

    n_items = len(asin2id)
    print(f"  ✓ 共 {n_items} 个物品")

    # 展示前5个映射
    print(f"  前 {N_SAMPLES} 个映射:")
    for i, (asin, iid) in enumerate(list(asin2id.items())[:N_SAMPLES]):
        print(f"    ASIN={asin}  →  item_id={iid}")

    return asin2id, n_items


def step2_check_metadata_matching(asin2id):
    """Step 2: 检查 meta_Books.jsonl 匹配情况"""
    print("\n" + "=" * 60)
    print("Step 2: 检查 meta_Books.jsonl 数据匹配")
    print("=" * 60)

    meta_path = os.path.join(DATA_DIR, 'meta_Books.jsonl')
    if not os.path.exists(meta_path):
        print(f"  ✗ 文件不存在: {meta_path}")
        return None, None

    item_texts = {}
    item_image_urls = {}
    total_count = 0
    matched_count = 0

    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            asin = item.get('parent_asin', '')
            if asin not in asin2id:
                continue

            item_id = asin2id[asin]
            matched_count += 1

            # 文本
            title = item.get('title', '')
            description = item.get('description', [])
            if isinstance(description, list):
                description = ' '.join(description)
            features_text = item.get('features', [])
            if isinstance(features_text, list):
                features_text = ' '.join(features_text)
            text = f"{title}. {description} {features_text}".strip()
            if not text or text == '.':
                text = title if title else "unknown item"
            item_texts[item_id] = text

            # 图片
            images = item.get('images', [])
            if images and isinstance(images, list):
                img_url = images[0].get('large', '')
                if img_url:
                    item_image_urls[item_id] = img_url

            # 到了调试数量就提前停止全量扫描的详细输出
            if matched_count <= N_SAMPLES:
                print(f"\n  --- 匹配样本 #{matched_count} ---")
                print(f"  ASIN: {asin} → item_id: {item_id}")
                print(f"  标题: {title[:80]}{'...' if len(title) > 80 else ''}")
                print(f"  文本长度: {len(text)} 字符")
                has_img = item_id in item_image_urls
                print(f"  图片URL: {'✓ 有' if has_img else '✗ 无'}")
                if has_img:
                    print(f"    {item_image_urls[item_id][:100]}")

    print(f"\n  汇总:")
    print(f"    jsonl 总条目:  {total_count}")
    print(f"    匹配成功:      {matched_count}/{len(asin2id)} ({100*matched_count/len(asin2id):.1f}%)")
    print(f"    有文本:        {len(item_texts)}")
    print(f"    有图片URL:     {len(item_image_urls)}")

    return item_texts, item_image_urls


def step3_test_bert(item_texts):
    """Step 3: 测试 BERT 文本编码器"""
    print("\n" + "=" * 60)
    print("Step 3: 测试 BERT 文本编码器 (bert-base-uncased)")
    print("=" * 60)

    try:
        from transformers import BertTokenizer, BertModel
    except ImportError:
        print("  ✗ transformers 未安装，请运行: pip install transformers")
        return False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    print("  加载 tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print("  加载 BERT 模型...")
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.to(device)
    bert_model.eval()
    print("  ✓ 模型加载完成")

    # 取前 N_SAMPLES 个有文本的样本
    sample_ids = sorted(item_texts.keys())[:N_SAMPLES]
    sample_texts = [item_texts[iid] for iid in sample_ids]

    print(f"\n  编码 {len(sample_texts)} 个文本样本...")
    with torch.no_grad():
        encoded = tokenizer(
            sample_texts,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        print(f"    input_ids shape:      {input_ids.shape}")
        print(f"    attention_mask shape:  {attention_mask.shape}")

        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        print(f"    输出 last_hidden_state: {outputs.last_hidden_state.shape}")
        print(f"    [CLS] embedding:        {cls_emb.shape}")

    # 打印每个样本的详情
    cls_np = cls_emb.cpu().numpy()
    for i, iid in enumerate(sample_ids):
        emb = cls_np[i]
        text_preview = sample_texts[i][:60]
        print(f"\n  item_id={iid}:")
        print(f"    文本: \"{text_preview}...\"")
        print(f"    embedding shape: {emb.shape}")
        print(f"    范数: {np.linalg.norm(emb):.4f}")
        print(f"    均值: {emb.mean():.6f}, 标准差: {emb.std():.6f}")
        print(f"    前5维: {emb[:5]}")

    # 检查嵌入是否不全为零
    all_nonzero = all(np.any(cls_np[i] != 0) for i in range(len(sample_ids)))
    print(f"\n  ✓ 所有样本 embedding 均非零向量: {all_nonzero}")

    # 释放显存
    del bert_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


def step4_test_vit(item_image_urls):
    """Step 4: 测试 ViT 视觉编码器"""
    print("\n" + "=" * 60)
    print("Step 4: 测试 ViT 视觉编码器 (vit_base_patch16_224)")
    print("=" * 60)

    try:
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from PIL import Image
        import requests
        from io import BytesIO
    except ImportError as e:
        print(f"  ✗ 缺少依赖: {e}")
        print("  请运行: pip install timm Pillow requests")
        return False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    print("  加载 ViT 模型...")
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    vit_model = vit_model.to(device)
    vit_model.eval()
    print("  ✓ 模型加载完成")

    config = resolve_data_config({}, model=vit_model)
    transform = create_transform(**config)
    print(f"  预处理配置: input_size={config.get('input_size')}, "
          f"mean={config.get('mean')}, std={config.get('std')}")

    # 取前 N_SAMPLES 个有图片的样本
    sample_ids = sorted(item_image_urls.keys())[:N_SAMPLES]

    print(f"\n  下载并编码 {len(sample_ids)} 张图片...")
    with torch.no_grad():
        for iid in sample_ids:
            url = item_image_urls[iid]
            print(f"\n  item_id={iid}:")
            print(f"    URL: {url[:100]}")
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert('RGB')
                print(f"    原始图片大小: {img.size}")

                img_tensor = transform(img).unsqueeze(0).to(device)
                print(f"    预处理后 tensor: {img_tensor.shape}")

                feat = vit_model(img_tensor).cpu().numpy()[0]
                print(f"    embedding shape: {feat.shape}")
                print(f"    范数: {np.linalg.norm(feat):.4f}")
                print(f"    均值: {feat.mean():.6f}, 标准差: {feat.std():.6f}")
                print(f"    前5维: {feat[:5]}")
                nonzero = np.any(feat != 0)
                print(f"    ✓ 非零向量: {nonzero}")
            except Exception as e:
                print(f"    ✗ 失败: {e}")

    del vit_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


if __name__ == '__main__':
    print("🔍 EditKG 多模态预处理调试")
    print(f"数据目录: {DATA_DIR}")
    print(f"调试样本数: {N_SAMPLES}\n")

    # Step 1
    result = step1_check_asin_mapping()
    if result is None:
        sys.exit(1)
    asin2id, n_items = result

    # Step 2
    item_texts, item_image_urls = step2_check_metadata_matching(asin2id)
    if item_texts is None:
        sys.exit(1)

    # Step 3
    if item_texts:
        step3_test_bert(item_texts)
    else:
        print("\n  跳过 BERT 测试：无匹配文本数据")

    # Step 4
    if item_image_urls:
        step4_test_vit(item_image_urls)
    else:
        print("\n  跳过 ViT 测试：无匹配图片URL")

    print("\n" + "=" * 60)
    print("调试完成！")
    print("=" * 60)
