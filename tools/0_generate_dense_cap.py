#!/usr/bin/env python3
"""
vllm_dense_caption_base64_nojson.py

说明：
- 把本地图片编码为 base64 并随请求发送给本地 vLLM 服务（OpenAI-compatible）。
- 不要求模型返回 JSON。模型只输出自然语言的 dense caption（建议尽量只返回一条描述）。
- 脚本负责后处理并把每张图片的描述保存为一个对象到同一个 JSON 文件（数组）。

用法示例：
# 单张图片
python vllm_dense_caption_base64_nojson.py --image /path/to/photo.jpg --out results.json

# 批量目录（递归）
python vllm_dense_caption_base64_nojson.py --dir ./photos --out results.json

参数：
--host, --port, --model, --image, --dir, --out, --max-tokens, --temp, --timeout, --sleep
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif")

def find_images_in_dir(dirpath: str) -> List[str]:
    images = []
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                images.append(os.path.join(root, f))
    return sorted(images)

def get_image_resolution(path: str) -> Optional[Dict[str, int]]:
    """获取图片分辨率，返回 {"height": int, "width": int}，失败返回 None。"""
    try:
        with Image.open(path) as img:
            width, height = img.size
            return {"height": height, "width": width}
    except Exception as e:
        print(f"[ERROR] 无法获取图片分辨率 {path}: {e}", file=sys.stderr)
        return None

def encode_image_to_base64(path: str) -> Tuple[Optional[str], Optional[str]]:
    """读取本地图片并返回 (b64_string, mime_type)。失败返回 (None, None)。"""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as e:
        print(f"[ERROR] 无法读取图片 {path}: {e}", file=sys.stderr)
        return None, None
    try:
        b64 = base64.b64encode(data).decode("ascii")
    except Exception as e:
        print(f"[ERROR] base64 编码失败 {path}: {e}", file=sys.stderr)
        return None, None
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    return b64, mime_type

def make_payload(model: str, image_b64: str, mime_type: str, instruction: str, max_tokens: int = 512, temperature: float = 0.0) -> dict:
    """
    构建 OpenAI-like multimodal payload。
    使用了 {"type":"image","image":{"b64":..., "mime_type":...}} 的字段约定。
    如你的服务期望别的字段（例如 image_url 或 data_uri），请调整此函数。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": {"b64": image_b64, "mime_type": mime_type}},
            ],
        }
    ]
    return {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }

def extract_text_from_content(content: Any) -> Optional[str]:
    """
    从多种可能的 content 结构中提取可读文本（优先返回字符串）。
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 找第一个显然是文本的元素
        for part in content:
            if isinstance(part, str):
                return part
            if isinstance(part, dict):
                for key in ("text", "content", "output_text"):
                    if key in part and isinstance(part[key], str):
                        return part[key]
        # 否则拼接为字符串
        try:
            return " ".join([p if isinstance(p, str) else json.dumps(p, ensure_ascii=False) for p in content])
        except Exception:
            return str(content)
    if isinstance(content, dict):
        for key in ("text", "content", "output_text"):
            if key in content and isinstance(content[key], str):
                return content[key]
        # 否则序列化整个 dict
        return json.dumps(content, ensure_ascii=False)
    return str(content)

def parse_model_response(resp_json: dict) -> Tuple[Optional[str], dict]:
    """
    从 OpenAI-like 响应中提取模型生成的文本（非 JSON），返回 (text_or_None, raw_resp).
    优先从 choices[0].message.content 或 choices[0].text 中提取。
    """
    try:
        choices = resp_json.get("choices", [])
        if not choices:
            return None, resp_json
        first = choices[0]
        # OpenAI-like: choices[0].message.content 可能为 str/list/dict
        message = first.get("message") or {}
        content = message.get("content")
        if content is None:
            # 备用位置
            content = first.get("text") or (message.get("text") if isinstance(message, dict) else None)
        text = extract_text_from_content(content)
        if text is not None:
            return text, resp_json
    except Exception as e:
        return None, {"error": str(e), "raw": resp_json}
    return None, resp_json

def clean_caption(text: str) -> str:
    """
    清理模型输出的文本：
    - 去除首尾空白
    - 去掉外部三引号或首尾引号（如果有）
    - 移除常见前缀如 "Caption:", "Answer:", "Response:" 等
    """
    if text is None:
        return ""
    s = text.strip()
    # 去除 ``` ``` 包围
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`").strip()
    # 去除首尾单/双引号
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # 去掉常见前缀
    for pref in ("Caption:", "caption:", "Answer:", "answer:", "Response:", "response:", "Description:", "description:"):
        if s.startswith(pref):
            s = s[len(pref):].strip()
            break
    return s

def process_single_image(host: str, port: int, model: str, image_path: str, instruction: str, timeout: int, max_tokens: int, temp: float) -> Tuple[str, List[str], Optional[Dict[str, int]]]:
    """
    处理单张图片：编码->请求->解析->清理，返回 (path, [caption], resolution)。
    若失败返回空 list 作为 cap，resolution 可能为 None。
    """
    # 获取图片分辨率
    resolution = get_image_resolution(image_path)

    b64, mime = encode_image_to_base64(image_path)
    if b64 is None:
        return image_path, [], resolution

    payload = make_payload(model, b64, mime, instruction, max_tokens=max_tokens, temperature=temp)
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        print(f"[ERROR] HTTP 请求失败: {e} (图片: {image_path})", file=sys.stderr)
        return image_path, [], resolution

    if resp.status_code != 200:
        print(f"[ERROR] 服务器返回状态码 {resp.status_code} (图片: {image_path})", file=sys.stderr)
        try:
            print(resp.text, file=sys.stderr)
        except Exception:
            pass
        return image_path, [], resolution

    try:
        resp_json = resp.json()
    except ValueError:
        print(f"[ERROR] 响应不是 JSON (图片: {image_path})，原始文本前1000字符：", file=sys.stderr)
        print(resp.text[:1000], file=sys.stderr)
        return image_path, [], resolution

    text, raw = parse_model_response(resp_json)
    if not text:
        print(f"[WARN] 未能提取文本（图片: {image_path}），详见 raw response 输出到 stderr", file=sys.stderr)
        print(json.dumps(raw, ensure_ascii=False)[:2000], file=sys.stderr)
        return image_path, [], resolution

    caption = clean_caption(text)
    if not caption:
        return image_path, [], resolution
    return image_path, [caption], resolution

def main():
    parser = argparse.ArgumentParser(description="Use local vLLM (base64 images) to get dense captions (model returns plain text), save all results in one JSON file.")
    parser.add_argument("--host", default="localhost", help="vLLM host (default localhost)")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port (default 8000)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default {DEFAULT_MODEL})")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Single image path to process")
    group.add_argument("--dir", help="Directory to scan for images (recursive)")
    parser.add_argument("--out", default="out.json", help="Output JSON file (single array) (default out.json)")
    parser.add_argument("--max-tokens", type=int, default=256, help="max completion tokens (default 256)")
    parser.add_argument("--temp", type=float, default=0.0, help="temperature (default 0.0)")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds (default 120)")
    parser.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between requests (default 0)")
    args = parser.parse_args()

    # 指令：不让模型返回 JSON，只返回一条描述文本（尽量简洁但密集）
    instruction = (
        "Provide a single, vivid, dense image description (one paragraph). "
        "Focus on posture, clothing, emotion, important objects, relationships, and the environment. "
        "Make the description informative and compact."
    )

    if args.image:
        images = [args.image]
    else:
        images = find_images_in_dir(args.dir)
        if not images:
            print(f"[ERROR] 在目录 {args.dir} 中未找到图片（后缀: {IMAGE_EXTS}）", file=sys.stderr)
            sys.exit(2)

    results = []
    total = len(images)
    for idx, img in enumerate(images, 1):
        print(f"[{idx}/{total}] 处理: {img}", flush=True)
        path, caps, resolution = process_single_image(args.host, args.port, args.model, img, instruction, args.timeout, args.max_tokens, args.temp)
        result_item = {"path": path, "cap": caps}
        if resolution is not None:
            result_item["resolution"] = resolution
        results.append(result_item)
        if args.sleep > 0:
            time.sleep(args.sleep)

    # 写入单一 JSON 数组文件
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[OK] 所有结果已保存至 {args.out}")
    except Exception as e:
        print(f"[ERROR] 保存结果失败: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
