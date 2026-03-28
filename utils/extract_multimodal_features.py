# -*- coding: utf-8 -*-
"""
==========================================================================
extract_multimodal_features.py — 多模态特征提取脚本
==========================================================================
功能说明：
    从 Amazon 数据集的 meta_Books.jsonl 中提取视觉和文本特征。
    
    流程：
    1. 读取 item_list.txt，建立 ASIN → 内部 item_id 的映射
    2. 读取 meta_Books.jsonl，提取每个 item 的图片 URL 和文字描述
    3. 使用 BERT (bert-base-uncased) 编码文本 → text_feat.npy
    4. 使用 ViT (vit_base_patch16_224) 编码图片 → image_feat.npy
    
    特征存储为 [n_items, feat_dim] 的 numpy 数组，
    行索引与 item_list.txt 中的 remap_id 对齐。

    参考 R2MR 的多模态编码方式：
    - 视觉编码器：Vision Transformer (ViT)
    - 文本编码器：BERT (bert-base-uncased)

使用方式：
    cd EditKG-main
    python utils/extract_multimodal_features.py --dataset amazon-book

依赖安装：
    pip install transformers timm Pillow requests
==========================================================================
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')


def load_asin_to_itemid(data_dir):
    """
    从 item_list.txt 读取 ASIN → 内部 item_id 的映射。
    
    item_list.txt 格式：
        org_id remap_id freebase_id
        0553092626 0 m.045wq1q
        ...
    
    返回:
        asin2id: {ASIN_str: int(remap_id)}
        n_items: 物品总数
    """
    item_list_path = os.path.join(data_dir, 'item_list.txt')
    asin2id = {}
    with open(item_list_path, 'r', encoding='utf-8') as f:
        header = f.readline()  # 跳过表头
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                parts = line.strip().split()
            if len(parts) >= 2:
                asin = parts[0]
                remap_id = int(parts[1])
                asin2id[asin] = remap_id
    n_items = len(asin2id)
    print(f"[item_list.txt] 共加载 {n_items} 个物品的 ASIN → item_id 映射")
    return asin2id, n_items


def load_metadata(data_dir, asin2id):
    """
    从 meta_Books.jsonl 中提取每个匹配物品的文本和图片 URL。
    
    文本 = title + " " + description (拼接)
    图片 = images[0]["large"] (第一张大图的URL)
    
    返回:
        item_texts: {item_id: text_str}
        item_image_urls: {item_id: url_str}
        matched_count: 匹配成功的物品数量
    """
    meta_path = os.path.join(data_dir, 'meta_Books.jsonl')
    item_texts = {}
    item_image_urls = {}
    matched_count = 0
    total_count = 0

    print(f"[meta_Books.jsonl] 正在读取元数据...")
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取 meta_Books.jsonl"):
            total_count += 1
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # 通过 parent_asin 匹配
            asin = item.get('parent_asin', '')
            if asin not in asin2id:
                continue

            item_id = asin2id[asin]
            matched_count += 1

            # 提取文本：title + description
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

            # 提取图片 URL
            images = item.get('images', [])
            if images and isinstance(images, list):
                img_url = images[0].get('large', '')
                if img_url:
                    item_image_urls[item_id] = img_url

    print(f"[meta_Books.jsonl] 共 {total_count} 条记录，匹配成功 {matched_count} 个物品")
    print(f"  有文本的物品: {len(item_texts)}, 有图片URL的物品: {len(item_image_urls)}")
    return item_texts, item_image_urls, matched_count


def extract_text_features(item_texts, n_items, device, max_len=64, batch_size=128):
    """
    使用 BERT (bert-base-uncased) 提取文本特征。
    
    参考 R2MR 的文本编码方式 (common/xbert.py + utils/dataset.py):
    - 使用 BertTokenizer 分词
    - 使用 BertModel 编码，取 [CLS] token 的输出作为文本表示
    
    参数:
        item_texts: {item_id: text_str}
        n_items: 物品总数
        device: torch.device
        max_len: 最大序列长度（R2MR 中为 32，这里用 64 以容纳更多信息）
        batch_size: 批次大小
    
    返回:
        text_feat: [n_items, 768] numpy 数组
    """
    from transformers import BertTokenizer, BertModel

    print("\n========== 文本特征提取 (BERT) ==========")
    print("加载 BERT 模型 (bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.to(device)
    bert_model.eval()

    feat_dim = 768  # BERT-base 的隐藏维度
    text_feat = np.zeros((n_items, feat_dim), dtype=np.float32)

    # 收集所有有文本的 item_id 列表
    item_ids = sorted(item_texts.keys())
    total = len(item_ids)
    print(f"待编码物品数: {total}/{n_items}")

    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc="BERT 编码文本"):
            batch_ids = item_ids[start:start + batch_size]
            batch_texts = [item_texts[iid] for iid in batch_ids]

            encoded = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # 取 [CLS] token 的输出（第0个位置），作为整个文本的表示
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            for j, iid in enumerate(batch_ids):
                text_feat[iid] = cls_embeddings[j]

    print(f"文本特征提取完成，形状: {text_feat.shape}")
    return text_feat


def _download_single_image(args_tuple):
    """下载单张图片并返回 (item_id, image_bytes) 或 (item_id, None)。"""
    import requests
    iid, url = args_tuple
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return (iid, response.content)
    except Exception:
        return (iid, None)


def extract_image_features(item_image_urls, n_items, device, batch_size=64, num_workers=24):
    """
    使用 ViT (vit_base_patch16_224) 提取图像特征。
    多线程并发下载图片，批量 ViT 编码。
    
    参数:
        item_image_urls: {item_id: url_str}
        n_items: 物品总数
        device: torch.device
        batch_size: ViT 编码批次大小
        num_workers: 并发下载线程数
    
    返回:
        image_feat: [n_items, 768] numpy 数组
    """
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from PIL import Image
    from io import BytesIO
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n========== 图像特征提取 (ViT) ==========")
    print("加载 ViT 模型 (vit_base_patch16_224)...")
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    vit_model = vit_model.to(device)
    vit_model.eval()

    # 获取 ViT 的预处理配置
    config = resolve_data_config({}, model=vit_model)
    transform = create_transform(**config)

    feat_dim = 768  # ViT-base 的输出维度
    image_feat = np.zeros((n_items, feat_dim), dtype=np.float32)

    item_ids = sorted(item_image_urls.keys())
    total = len(item_ids)
    print(f"待编码物品数: {total}/{n_items}")
    print(f"并发下载线程数: {num_workers}")

    success_count = 0
    fail_count = 0

    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc="ViT 编码图片"):
            batch_ids = item_ids[start:start + batch_size]

            # ---- 多线程并发下载本批次图片 ----
            download_args = [(iid, item_image_urls[iid]) for iid in batch_ids]
            downloaded = {}  # iid -> bytes
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_download_single_image, arg): arg[0]
                           for arg in download_args}
                for future in as_completed(futures):
                    iid, content = future.result()
                    if content is not None:
                        downloaded[iid] = content

            # ---- 解码 + 预处理 ----
            batch_images = []
            valid_ids = []
            for iid in batch_ids:
                if iid not in downloaded:
                    fail_count += 1
                    if fail_count <= 10:
                        print(f"  [图片失败] item_id={iid}: 下载失败")
                    continue
                try:
                    img = Image.open(BytesIO(downloaded[iid])).convert('RGB')
                    img_tensor = transform(img)
                    batch_images.append(img_tensor)
                    valid_ids.append(iid)
                    success_count += 1
                except Exception as e:
                    fail_count += 1
                    if fail_count <= 10:
                        print(f"  [图片失败] item_id={iid}: {type(e).__name__}: {e}")

            if not batch_images:
                continue

            batch_tensor = torch.stack(batch_images).to(device)
            features = vit_model(batch_tensor).cpu().numpy()

            for j, iid in enumerate(valid_ids):
                image_feat[iid] = features[j]

    print(f"图像特征提取完成：成功 {success_count}，失败 {fail_count}")
    print(f"图像特征形状: {image_feat.shape}")
    return image_feat


def main():
    parser = argparse.ArgumentParser(description="多模态特征提取")
    parser.add_argument("--dataset", type=str, default="amazon-book", help="数据集名称")
    parser.add_argument("--data_path", type=str, default="data/", help="数据根目录")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--text_batch_size", type=int, default=128, help="文本编码批次大小")
    parser.add_argument("--image_batch_size", type=int, default=32, help="图像编码批次大小")
    parser.add_argument("--num_workers", type=int, default=24, help="图像下载并发线程数")
    parser.add_argument("--max_text_len", type=int, default=64, help="最大文本长度")
    parser.add_argument("--skip_images", action="store_true", help="跳过图像特征提取（仅提取文本）")
    parser.add_argument("--force_reextract", action="store_true", help="强制重新提取，忽略已有的 .npy 文件")
    parser.add_argument("--debug", action="store_true", help="调试模式：只处理少量样本，打印详细信息")
    parser.add_argument("--debug_samples", type=int, default=5, help="调试模式下每种模态处理的样本数")
    args = parser.parse_args()

    data_dir = os.path.join(args.data_path, args.dataset)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # Step 1: 加载 ASIN → item_id 映射
    asin2id, n_items = load_asin_to_itemid(data_dir)

    # Step 2: 加载元数据
    item_texts, item_image_urls, matched = load_metadata(data_dir, asin2id)

    # ===== 调试模式：只保留少量样本 =====
    if args.debug:
        print(f"\n{'='*60}")
        print(f"[DEBUG] 调试模式已开启，每种模态只处理 {args.debug_samples} 个样本")
        print(f"{'='*60}")

        # 只保留前 debug_samples 个有文本的样本
        debug_text_ids = sorted(item_texts.keys())[:args.debug_samples]
        item_texts = {k: item_texts[k] for k in debug_text_ids}
        print(f"\n[DEBUG] 文本样本 ({len(item_texts)} 个):")
        for iid in debug_text_ids:
            txt = item_texts[iid]
            print(f"  item_id={iid}: \"{txt[:100]}{'...' if len(txt)>100 else ''}\"  (长度={len(txt)})")

        # 只保留前 debug_samples 个有图片的样本
        debug_img_ids = sorted(item_image_urls.keys())[:args.debug_samples]
        item_image_urls = {k: item_image_urls[k] for k in debug_img_ids}
        print(f"\n[DEBUG] 图片样本 ({len(item_image_urls)} 个):")
        for iid in debug_img_ids:
            print(f"  item_id={iid}: {item_image_urls[iid][:120]}")

    # Step 3: 提取文本特征 (BERT)
    text_feat_path = os.path.join(data_dir, 'text_feat.npy')
    if not args.force_reextract and os.path.isfile(text_feat_path):
        text_feat = np.load(text_feat_path, allow_pickle=True)
        print(f"\n[跳过] 文本特征已存在: {text_feat_path}, 形状: {text_feat.shape}（使用 --force_reextract 强制重新提取）")
    else:
        text_feat = extract_text_features(
            item_texts, n_items, device,
            max_len=args.max_text_len,
            batch_size=args.text_batch_size
        )
        np.save(text_feat_path, text_feat)
        print(f"✓ 文本特征已保存: {text_feat_path}")

    if args.debug:
        print(f"\n[DEBUG] ===== BERT 文本特征质量检查 =====")
        for iid in sorted(item_texts.keys())[:args.debug_samples]:
            emb = text_feat[iid]
            norm = np.linalg.norm(emb)
            print(f"  item_id={iid}: 范数={norm:.4f}, 均值={emb.mean():.6f}, "
                  f"标准差={emb.std():.6f}, 非零={np.count_nonzero(emb)}/{len(emb)}")
            print(f"    前10维: {emb[:10]}")
        nonzero_cnt = np.sum(np.any(text_feat != 0, axis=1))
        print(f"  有效文本特征行数: {nonzero_cnt}/{n_items}")
    # Step 4: 提取图像特征 (ViT)
    image_feat_path = os.path.join(data_dir, 'image_feat.npy')
    if not args.skip_images:
        if not args.force_reextract and os.path.isfile(image_feat_path):
            image_feat = np.load(image_feat_path, allow_pickle=True)
            print(f"\n[跳过] 图像特征已存在: {image_feat_path}, 形状: {image_feat.shape}（使用 --force_reextract 强制重新提取）")
        else:
            image_feat = extract_image_features(
                item_image_urls, n_items, device,
                batch_size=args.image_batch_size,
                num_workers=args.num_workers
            )
            np.save(image_feat_path, image_feat)
            print(f"✓ 图像特征已保存: {image_feat_path}")

        if args.debug:
            print(f"\n[DEBUG] ===== ViT 图像特征质量检查 =====")
            for iid in sorted(item_image_urls.keys())[:args.debug_samples]:
                emb = image_feat[iid]
                norm = np.linalg.norm(emb)
                print(f"  item_id={iid}: 范数={norm:.4f}, 均值={emb.mean():.6f}, "
                      f"标准差={emb.std():.6f}, 非零={np.count_nonzero(emb)}/{len(emb)}")
                print(f"    前10维: {emb[:10]}")
            nonzero_cnt = np.sum(np.any(image_feat != 0, axis=1))
            print(f"  有效图像特征行数: {nonzero_cnt}/{n_items}")


    else:
        print("已跳过图像特征提取")

    # 统计信息
    print("\n========== 特征提取汇总 ==========")
    print(f"数据集: {args.dataset}")
    print(f"物品总数: {n_items}")
    print(f"ASIN 匹配成功: {matched}/{n_items}")
    print(f"文本特征: {text_feat.shape} → {text_feat_path}")
    if not args.skip_images:
        print(f"图像特征: {image_feat.shape} → {image_feat_path}")
    
    # 统计有效特征比例
    text_nonzero = np.sum(np.any(text_feat != 0, axis=1))
    print(f"有效文本特征: {text_nonzero}/{n_items} ({100*text_nonzero/n_items:.1f}%)")
    if not args.skip_images:
        img_nonzero = np.sum(np.any(image_feat != 0, axis=1))
        print(f"有效图像特征: {img_nonzero}/{n_items} ({100*img_nonzero/n_items:.1f}%)")


if __name__ == '__main__':
    main()
