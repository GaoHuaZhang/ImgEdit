#!/usr/bin/env python3
"""
计算分割结果的 clip_score 和 aes_score

该脚本读取 4_yolo_seg.py 生成的分割结果 JSON 文件，为每个分割区域计算：
1. clip_score: 使用 CLIP 模型计算分割区域图像与类别名称的语义相似度
2. aes_score: 使用 aesthetic predictor 计算分割区域的审美分数

依赖安装:
    # 方式1: 使用原始 clip 库
    pip install torch torchvision
    pip install git+https://github.com/openai/CLIP.git
    pip install pycocotools pillow tqdm numpy

    # 方式2: 使用 transformers 库
    pip install torch torchvision transformers
    pip install pycocotools pillow tqdm numpy

    # AES 模型需要根据实际使用的模型安装相应依赖
    # 例如 improved-aesthetic-predictor:
    # pip install timm  # 如果使用 timm 模型

用法示例:
    # 使用原始 clip 库
    python 4.5_compute_scores.py \
        --json_folder ./output \
        --image_folder ./images \
        --clip_model ViT-B/32 \
        --aes_model_path ./aesthetic_predictor.pth \
        --batch_size 8

    # 使用 transformers 库
    python 4.5_compute_scores.py \
        --json_folder ./output \
        --image_folder ./images \
        --clip_model openai/clip-vit-base-patch32 \
        --aes_model_path ./aesthetic_predictor.pth \
        --batch_size 8

    # 如果不需要 AES 分数，可以省略 --aes_model_path
    python 4.5_compute_scores.py \
        --json_folder ./output \
        --image_folder ./images \
        --clip_model ViT-B/32
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pycocotools.mask as mask_util
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 尝试导入 CLIP 相关库
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    try:
        from transformers import CLIPProcessor, CLIPModel
        CLIP_AVAILABLE = True
        USE_TRANSFORMERS = True
    except ImportError:
        CLIP_AVAILABLE = False
        USE_TRANSFORMERS = False
        print("Warning: CLIP not available. Please install 'clip' or 'transformers'", file=sys.stderr)


def rle_to_mask(rle_str: str, height: int, width: int) -> np.ndarray:
    """将 RLE 字符串解码为 mask 数组"""
    rle = {"counts": rle_str.encode("utf-8"), "size": [height, width]}
    mask = mask_util.decode(rle)
    return mask.astype(bool)


def extract_region_from_mask(image: Image.Image, mask: np.ndarray, bbox: List[float]) -> Image.Image:
    """
    从原图中提取分割区域

    Args:
        image: 原始 PIL 图像
        mask: 布尔型 mask 数组
        bbox: [x1, y1, x2, y2] 格式的边界框

    Returns:
        提取的区域图像（白色背景）
    """
    # 将图像转换为 numpy 数组
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]

    # 确保 mask 尺寸匹配
    if mask.shape != (h, w):
        # 如果 mask 尺寸不匹配，需要调整
        from PIL import Image as PILImage
        mask_img = PILImage.fromarray((mask * 255).astype(np.uint8)).resize((w, h))
        mask = np.array(mask_img) > 127

    # 创建白色背景
    result = np.ones((h, w, 3), dtype=np.uint8) * 255

    # 将 mask 区域复制到结果中
    result[mask] = img_array[mask]

    # 裁剪到 bbox 区域（如果 bbox 有效）
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 > x1 and y2 > y1:
        result = result[y1:y2, x1:x2]

    return Image.fromarray(result)


class SegmentationDataset(Dataset):
    """用于批量处理分割区域的数据集"""

    def __init__(self, annotations: List[Dict], image_path: str, resolution: Dict[str, int]):
        self.annotations = annotations
        self.image_path = image_path
        self.resolution = resolution
        self.image = None

        # 加载图像
        if os.path.exists(image_path):
            self.image = Image.open(image_path).convert("RGB")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        if self.image is None:
            return None, None, ann

        # 解码 mask
        mask = rle_to_mask(
            ann["mask"],
            self.resolution["height"],
            self.resolution["width"]
        )

        # 提取区域
        region = extract_region_from_mask(
            self.image,
            mask,
            ann["bbox"]
        )

        return region, ann["class_name"], ann

    def get_image(self):
        return self.image


def compute_clip_score_batch(regions: List[Image.Image], texts: List[str],
                             clip_model, clip_preprocess, device) -> List[float]:
    """
    批量计算 CLIP 分数

    Args:
        regions: 分割区域图像列表
        texts: 类别名称列表
        clip_model: CLIP 模型
        clip_preprocess: CLIP 预处理函数
        device: 计算设备

    Returns:
        CLIP 分数列表
    """
    if not CLIP_AVAILABLE or clip_model is None:
        return [0.5] * len(regions)

    scores = []

    try:
        if USE_TRANSFORMERS:
            # 使用 transformers 库
            processor = clip_preprocess
            model = clip_model

            # 处理图像和文本
            inputs = processor(images=regions, text=texts, return_tensors="pt", padding=True).to(device)

            # 获取特征
            with torch.no_grad():
                outputs = model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

                # 归一化
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # 计算相似度（余弦相似度）
                similarity = (image_features * text_features).sum(dim=-1)
                scores = similarity.cpu().tolist()
        else:
            # 使用原始 clip 库
            # 预处理图像
            image_tensors = torch.stack([clip_preprocess(region) for region in regions]).to(device)
            text_tokens = clip.tokenize(texts, truncate=True).to(device)

            # 计算特征
            with torch.no_grad():
                image_features = clip_model.encode_image(image_tensors)
                text_features = clip_model.encode_text(text_tokens)

                # 归一化
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # 计算相似度（余弦相似度）
                similarity = (image_features * text_features).sum(dim=-1)
                scores = similarity.cpu().tolist()

    except Exception as e:
        print(f"Error computing CLIP scores: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        scores = [0.5] * len(regions)

    return scores


def compute_aes_score_batch(regions: List[Image.Image], aes_model, aes_preprocess, device) -> List[float]:
    """
    批量计算 Aesthetic 分数

    Args:
        regions: 分割区域图像列表
        aes_model: Aesthetic predictor 模型
        aes_preprocess: 预处理函数
        device: 计算设备

    Returns:
        Aesthetic 分数列表
    """
    scores = []

    try:
        # 预处理图像
        image_tensors = torch.stack([aes_preprocess(region) for region in regions]).to(device)

        # 计算分数
        with torch.no_grad():
            predictions = aes_model(image_tensors)
            # 假设模型输出是 [batch_size, 1] 或 [batch_size]
            if predictions.dim() > 1:
                predictions = predictions.squeeze(1)
            scores = predictions.cpu().tolist()

    except Exception as e:
        print(f"Error computing AES scores: {e}", file=sys.stderr)
        # 默认返回中等分数
        scores = [5.0] * len(regions)

    return scores


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
    """加载 CLIP 模型"""
    if not CLIP_AVAILABLE:
        return None, None

    try:
        if USE_TRANSFORMERS:
            model = CLIPModel.from_pretrained(model_name).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)
            model.eval()
            return model, processor
        else:
            # 对于原始 clip 库，model_name 应该是预训练模型名称
            # 如 "ViT-B/32", "ViT-B/16", "ViT-L/14" 等
            available_models = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", "RN50x4", "RN50x16"]
            if model_name not in available_models:
                # 如果不在列表中，尝试使用 ViT-B/32
                print(f"Warning: {model_name} not in available models, using ViT-B/32", file=sys.stderr)
                model_name = "ViT-B/32"
            model, preprocess = clip.load(model_name, device=device)
            model.eval()
            return model, preprocess
    except Exception as e:
        print(f"Error loading CLIP model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None, None


def load_aes_model(model_path: str, device: str = "cuda"):
    """
    加载 Aesthetic Predictor 模型

    支持 improved-aesthetic-predictor 格式的模型
    模型应该是一个 PyTorch 模型，接受图像张量输入，输出单个分数值
    """
    try:
        if not os.path.exists(model_path):
            print(f"Warning: AES model not found at {model_path}, using default scores", file=sys.stderr)
            return None, None

        # 尝试加载模型
        # 支持多种加载方式
        checkpoint = torch.load(model_path, map_location=device)

        # 如果 checkpoint 是字典，尝试提取模型
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # 需要根据实际模型结构创建模型
                # 这里提供一个通用的 ResNet 示例
                try:
                    import torchvision.models as models
                    model = models.resnet50(pretrained=False)
                    model.fc = torch.nn.Linear(model.fc.in_features, 1)
                    model.load_state_dict(checkpoint['state_dict'])
                except:
                    print("Warning: Could not load model from state_dict, using default scores", file=sys.stderr)
                    return None, None
            else:
                # 尝试直接使用整个字典作为模型（如果它是模型本身）
                model = checkpoint
        else:
            model = checkpoint

        # 确保是模型对象
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
            model.eval()

            # 标准 ImageNet 预处理
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            return model, preprocess
        else:
            print("Warning: Loaded object is not a PyTorch model", file=sys.stderr)
            return None, None

    except Exception as e:
        print(f"Error loading AES model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None, None


def process_single_json(json_path: str, image_folder: str, clip_model, clip_preprocess,
                       aes_model, aes_preprocess, device: str, batch_size: int = 8):
    """
    处理单个 JSON 文件，计算所有分割区域的 clip_score 和 aes_score
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}", file=sys.stderr)
        return False

    # 检查是否有 segmentation 字段
    if "segmentation" not in data:
        print(f"Warning: {json_path} has no 'segmentation' field", file=sys.stderr)
        return False

    segmentation = data["segmentation"]
    image_path = os.path.join(image_folder, data["path"])
    resolution = data.get("resolution", {"height": 1024, "width": 1024})

    # 收集所有需要处理的标注
    all_annotations = []
    annotation_keys = []  # 记录每个标注属于哪个列表和索引

    for seg_type in ["background", "object"]:
        if seg_type in segmentation:
            for idx, ann in enumerate(segmentation[seg_type]):
                # 检查是否已经有分数（避免重复计算）
                if "clip_score" not in ann or "aes_score" not in ann:
                    all_annotations.append(ann)
                    annotation_keys.append((seg_type, idx))

    if not all_annotations:
        # 所有分数都已计算
        return True

    # 创建数据集
    dataset = SegmentationDataset(all_annotations, image_path, resolution)

    if dataset.image is None:
        print(f"Warning: Image not found: {image_path}", file=sys.stderr)
        return False

    # 批量处理
    regions = []
    texts = []
    key_mapping = []

    for i in range(len(dataset)):
        region, text, ann = dataset[i]
        if region is not None:
            regions.append(region)
            texts.append(text)
            key_mapping.append(annotation_keys[i])

    if not regions:
        return False

    # 计算 CLIP 分数
    if clip_model is not None:
        clip_scores = compute_clip_score_batch(regions, texts, clip_model, clip_preprocess, device)
    else:
        clip_scores = [0.5] * len(regions)

    # 计算 AES 分数
    if aes_model is not None:
        aes_scores = compute_aes_score_batch(regions, aes_model, aes_preprocess, device)
    else:
        aes_scores = [5.0] * len(regions)

    # 更新 JSON 数据
    for (seg_type, idx), clip_score, aes_score in zip(key_mapping, clip_scores, aes_scores):
        segmentation[seg_type][idx]["clip_score"] = float(clip_score)
        segmentation[seg_type][idx]["aes_score"] = float(aes_score)

    # 保存更新后的 JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving {json_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="计算分割结果的 clip_score 和 aes_score"
    )
    parser.add_argument(
        "--json_folder",
        type=str,
        required=True,
        help="包含分割结果 JSON 文件的文件夹"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="原始图片文件夹"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="输出文件夹（如果为 None，则原地更新 JSON 文件）"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32" if not USE_TRANSFORMERS else "openai/clip-vit-base-patch32",
        help="CLIP 模型名称（clip库: ViT-B/32, ViT-B/16等; transformers: openai/clip-vit-base-patch32等）"
    )
    parser.add_argument(
        "--aes_model_path",
        type=str,
        default=None,
        help="Aesthetic predictor 模型路径（可选）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批处理大小"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行处理的 worker 数量"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备 (cuda/cpu)"
    )

    args = parser.parse_args()

    # 设置设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU", file=sys.stderr)
        device = "cpu"

    # 加载模型
    print("Loading CLIP model...")
    clip_model, clip_preprocess = load_clip_model(args.clip_model, device)
    if clip_model is None:
        print("Warning: CLIP model not loaded, clip_score will be set to 0.5", file=sys.stderr)

    print("Loading AES model...")
    aes_model, aes_preprocess = None, None
    if args.aes_model_path:
        aes_model, aes_preprocess = load_aes_model(args.aes_model_path, device)
        if aes_model is None:
            print("Warning: AES model not loaded, aes_score will be set to 5.0", file=sys.stderr)
    else:
        print("Warning: No AES model path provided, aes_score will be set to 5.0", file=sys.stderr)

    # 查找所有 JSON 文件
    json_files = []
    for root, dirs, files in os.walk(args.json_folder):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    if not json_files:
        print(f"No JSON files found in {args.json_folder}", file=sys.stderr)
        return

    print(f"Found {len(json_files)} JSON files")

    # 处理文件
    output_folder = args.output_folder or args.json_folder
    os.makedirs(output_folder, exist_ok=True)

    success_count = 0
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        # 确定输出路径
        if args.output_folder:
            rel_path = os.path.relpath(json_file, args.json_folder)
            output_path = os.path.join(output_folder, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = json_file

        # 如果输出路径不同，先复制文件
        if output_path != json_file:
            import shutil
            shutil.copy2(json_file, output_path)

        # 处理文件
        if process_single_json(
            output_path,
            args.image_folder,
            clip_model,
            clip_preprocess,
            aes_model,
            aes_preprocess,
            device,
            args.batch_size
        ):
            success_count += 1

    print(f"\nCompleted! Successfully processed {success_count}/{len(json_files)} files")


if __name__ == "__main__":
    main()

