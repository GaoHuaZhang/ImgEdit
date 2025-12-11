import base64
import os
import json
import argparse
from multiprocessing import Pool, cpu_count
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from openai import OpenAI


prompt = """
You are a data rater specializing in generate image editing prompt. Given two images of the same person—before and after editing—write a concise prompt that describes only the change in the subject's action, pose, or facial expression. Do not mention any environmental or background details. For example: 'A man swings a golf club.'
Below are the images before and after editing:
"""



# Function to convert an image file to Base64
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

# Retry decorator with exponential backoff for call_gpt
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100))
def call_gpt(original_image_path, result_image_path, client, model_name):
    try:
        # Convert images to Base64 encoding
        original_image_base64 = image_to_base64(original_image_path)
        result_image_base64 = image_to_base64(result_image_path)

        if not original_image_base64 or not result_image_base64:
            return {"error": "Image conversion failed"}

        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{original_image_base64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{result_image_base64}"},
                    }
                ]
            }],
            max_tokens=1024,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in calling vLLM API: {e}")
        raise  # Reraise the exception to trigger a retry


def process_json(args):
    """
    单个 json 文件的处理函数。
    注意：必须放在顶层（不能嵌套在函数里），
    否则 multiprocessing 在 Windows 下会 pickling 失败。
    """
    json_path, image_folder, base_url, api_key, model_name = args
    # 在每个进程中创建client，因为client对象不能被pickle
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] 解析 {json_path} 失败: {e}")
        return None

    # 取出 start_frame / end_frame 相对路径
    meta = data.get("metadata", {}).get("image_paths", {})
    start_rel = meta.get("start_frame")
    end_rel   = meta.get("end_frame")

    if not (start_rel and end_rel):
        print(f"[WARN] {json_path} 缺少 start_frame / end_frame 字段")
        return None

    start_abs = os.path.join(image_folder, start_rel)
    end_abs   = os.path.join(image_folder, end_rel)

    if not (os.path.exists(start_abs) and os.path.exists(end_abs)):
        print(f"[WARN] {json_path} 对应的图片不存在")
        return None

    # 调用 vLLM API
    try:
        response = call_gpt(start_abs, end_abs, client, model_name)
        if isinstance(response, dict) and "error" in response:
            print(f"[ERROR] {json_path} 处理失败: {response['error']}")
            return None
        data["prompt"] = response
    except Exception as e:
        print(f"[ERROR] {json_path} 调用API失败: {e}")
        return None

    # 回写 json
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return json_path
    except Exception as e:
        print(f"[ERROR] 写回 {json_path} 失败: {e}")
        return None


def process_directory_parallel(json_folder: str,
                               image_folder: str,
                               base_url,
                               api_key,
                               model_name,
                               num_processes: int | None = None):
    """
    并行遍历并处理 json 文件
    """
    json_files = []
    for root, dirs, files in os.walk(json_folder):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    if not json_files:
        print("未找到任何 json 文件")
        return

    num_processes = num_processes or cpu_count()
    params = [(json_file, image_folder, base_url, api_key, model_name) for json_file in json_files]

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_json, params),
                  total=len(params),
                  desc="Processing jsons"))



# Main function to handle argument parsing and initiate processing
def main():
    parser = argparse.ArgumentParser(description="Process image editing tasks in a directory")
    parser.add_argument('--json_folder', type=str, required=True, help="Path to the folder containing JSON files")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the images containing subfolders with image editing tasks")
    parser.add_argument('--num_processes', type=int, default=None, help="Number of processes to use for parallel processing (default: CPU count)")
    parser.add_argument('--base_url', type=str, default="http://localhost:8000/v1", help="vLLM service base URL")
    parser.add_argument('--api_key', type=str, default="EMPTY", help="API key for vLLM service (default: EMPTY)")
    parser.add_argument('--model_name', type=str, default="gpt-4o", help="Model name for vLLM service")

    args = parser.parse_args()

    # Process the directory with the specified number of processes
    process_directory_parallel(
        args.json_folder,
        args.image_folder,
        args.base_url,
        args.api_key,
        args.model_name,
        args.num_processes
    )

    logging.info("Processing complete.")
if __name__ == "__main__":
    main()
