import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


prompt_format = (
    "Given an image caption, please retrieve the entity words that indicate background and visually separable objects and summarize it. "
    "[Definition of background] The background spaces that appear in most of the image area. "
    "[Definition of object] Entities that are visually separable, tangible, and physically present in part of the image, including object, human and animals. "
    "Attention! All entity words need to strictly follow the rules below: "
    "1) The entity word is a singular or plural noun without any quantifier or descriptive phrase. "
    "2) The entity word must be an exact subset of the caption, including its characters, words, and symbols. (e.g, 'red top' better than 'top', 'martial arts uniforms' better than 'uniforms') "
    "3) Exclude abstract or non-physical concepts (e.g., 'facial expressions', 'gestures', 'stance', 'details'). "
    "4) Exclude actions or descriptions (e.g., 'adjusting', 'imitating'). "
    "5) Do not modify or interpret any part of the caption. "
    "6) Keep the summary within 15 words, focusing on object, background, and their interaction (e.g., action). "
    "Here is an example, follow this format to output the results:"
    "Caption: On a busy city street, with soft winter light filtering through overcast skies, a woman in a mask and heavy coat pauses beside a storefront. Her long brown hair moves slightly in the breeze as she holds up a small green-capped bottle, gesturing like she's offering a product review or giving instructions."
    "Output: {'background': ['street', 'sky'], 'object': ['woman', 'mask', 'coat', 'long brown hair', 'green-capped bottle'], 'summary': ['Woman presents bottle on city street in winter light.']}"
    "Caption: {{{}}}"
    "Output: "
)

def extract_data_from_response(response):
    # Convert all double quotes to single quotes at the beginning
    response = response.replace('"', "'")

    # Regex to match the structure, supporting single quotes for keys and values
    background_pattern = r"'background'\s*:\s*\[(.*?)\]"
    object_pattern = r"'object'\s*:\s*\[(.*?)\]"
    summary_pattern = r"'summary'\s*:\s*\[(.*?)\]"

    # Extracting background, object, and summary using regex
    background_match = re.search(background_pattern, response)
    object_match = re.search(object_pattern, response)
    summary_match = re.search(summary_pattern, response)

    # Initialize the result dictionary
    result = {"background": [], "object": [], "summary": []}

    if background_match:
        # Split and process background items
        background_str = background_match.group(1).strip()
        if background_str:
            background_items = [item.strip("' \"") for item in background_str.split("', '")]
            result["background"] = background_items

    if object_match:
        # Split and process object items
        object_str = object_match.group(1).strip()
        if object_str:
            object_items = [item.strip("' \"") for item in object_str.split("', '")]
            result["object"] = object_items

    if summary_match:
        # Process summary items (it's a list with one item)
        summary_str = summary_match.group(1).strip()
        if summary_str:
            summary_items = [item.strip("' \"") for item in summary_str.split("', '")]
            result["summary"] = summary_items

    # Check if all parts were successfully matched
    if background_match and object_match and summary_match:
        flag = True
    else:
        flag = False

    return result, flag


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100))
def call_gpt(prompt, model_name, api_key, base_url):
    client = OpenAI(
        api_key=api_key or "dummy-key",  # vLLM通常不需要真实的API key
        base_url=base_url
    )
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content


def process_file(json_data, model_name, api_key, base_url, output_folder):
    key = json_data[0].replace("/", "").replace(".jpg", "")
    json_path = f"{key}.json"
    dump_data = {}
    image_metadata = json_data[1]
    input_prompt = prompt_format.replace("{{{}}}", image_metadata["cap"][0].strip()).replace("\n", "").replace("\t", "").strip()

    response = call_gpt(input_prompt, model_name=model_name, api_key=api_key, base_url=base_url)

    try:
        response_data = json.loads(response)
    except json.JSONDecodeError:
        response_data, flag = extract_data_from_response(response)
        if not flag:
            print(f"The response is not in valid format, skipping. Key: {key}")
            return

    image_metadata["tags"] = response_data
    dump_data[key] = image_metadata
    with open(os.path.join(output_folder, os.path.basename(json_path)), "w") as f:
        json.dump(dump_data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some input and output folders with model settings.")
    parser.add_argument(
        "--input_json", type=str, required=True, help="Path to the caption json."
    )
    parser.add_argument(
        "--output_json_folder", type=str, required=True, help="Path to the output json folder."
    )
    parser.add_argument("--num_worker", type=int, default=32, help="Number of threads for parallel processing")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL for vLLM service (e.g., http://localhost:8000/v1)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for vLLM service")
    parser.add_argument("--api_key", type=str, default="dummy-key", help="API key (usually not needed for local vLLM)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Config
    num_worker = args.num_worker
    input_json = args.input_json
    output_folder = args.output_json_folder
    model_name = args.model_name
    api_key = args.api_key
    base_url = args.base_url

    with open(input_json, 'r', encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    fflag = True
    for _ in range(3):
        json_datas = []
        for item in data:
            key = item['path'].replace("/", "").replace(".jpg", "")
            json_path = os.path.join(output_folder, f"{key}.json")
            if not os.path.exists(json_path):
                json_datas.append((key, item))

        if not json_datas:
            fflag = False
            break

        with ThreadPoolExecutor(max_workers=num_worker) as executor:
            futures = []
            with tqdm(total=len(json_datas), desc="Processing JSON files") as pbar:
                for json_data in json_datas:
                    future = executor.submit(process_file, json_data, model_name, api_key, base_url, output_folder)
                    futures.append(future)

                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

    if fflag:
        print("Processing half completed!")
    else:
        print("Processing full completed!")

