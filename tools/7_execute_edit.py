#!/usr/bin/env python3
"""
步骤5: 执行图片编辑

根据不同的编辑类型（add/remove/replace/adjust等）采用相应的编辑策略，
调用inpaint-workflow中的脚本进行图片编辑。

支持的编辑类型：
- 单轮编辑: add, remove, replace, adjust, background, extract
- 多轮编辑: hybrid, content_memory, content_understand, version_backtrack

使用方法:
    python 6_execute_edit.py \\
        --json_dir /path/to/step4/output \\
        --base_img_dir /path/to/original/images \\
        --result_dir /path/to/edit/results \\
        [--workflow_dir /path/to/inpaint-workflow] \\
        [--edit_types add remove replace] \\
        [--no-dlc]

环境变量要求（使用DLC分布式执行时）:
    - RANK: 当前节点的rank
    - MASTER_ADDR: master节点地址
    - MASTER_PORT: master节点端口
    - WORLD_SIZE: 总节点数
    - PYTHONPATH: Python解释器路径
    - COMFYUI_PATH: ComfyUI安装路径

示例:
    # 处理所有编辑类型
    python 6_execute_edit.py \\
        --json_dir ./data/step4_output \\
        --base_img_dir ./data/images \\
        --result_dir ./data/edit_results

    # 只处理add和remove类型
    python 6_execute_edit.py \\
        --json_dir ./data/step4_output \\
        --base_img_dir ./data/images \\
        --result_dir ./data/edit_results \\
        --edit_types add remove
"""

import json
import os
import sys
import argparse
import glob
from pathlib import Path
from typing import Dict, List
import subprocess
from tqdm import tqdm

# 编辑类型到脚本的映射
EDIT_TYPE_TO_SCRIPT = {
    "add": "dlc_add.py",
    "remove": "dlc_remove.py",
    "replace": "dlc_replace.py",
    "adjust": "dlc_adjust_canny.py",
    "background": "dlc_background_change.py",
    "extract": "dlc_extract_ref.py",
    # 混合编辑类型
    "hybrid": {
        "add": "dlc_compose_add.py",
        "remove": "dlc_compose_remove.py",
        "replace": "dlc_compose_replace.py",
        "adjust": "dlc_compose_adjust.py",
    },
    # 多轮编辑类型
    "content_memory": {
        "add": "dlc_compose_add_omit.py",
        "remove": "dlc_compose_remove_omit.py",
        "replace": "dlc_compose_replace_omit.py",
        "adjust": "dlc_compose_adjust_omit.py",
    },
    "content_understand": {
        "add": "dlc_compose_add_omit.py",
        "remove": "dlc_compose_remove_omit.py",
        "replace": "dlc_compose_replace_omit.py",
        "adjust": "dlc_compose_adjust_omit.py",
    },
    "version_backtrack": {
        "replace": "dlc_compose_replace_version.py",
        "adjust": "dlc_compose_adjust_version.py",
    },
}


def load_json_file(json_path: str) -> dict:
    """加载JSON文件"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
        return None


def get_edit_type(data: dict) -> str:
    """从数据中提取编辑类型"""
    edit_type = data.get("edit_type")
    if isinstance(edit_type, list):
        # 对于多轮编辑，返回主类型
        if len(edit_type) > 0:
            return edit_type[0]
    return edit_type if isinstance(edit_type, str) else None


def classify_tasks_by_type(json_dir: str) -> Dict[str, List[str]]:
    """
    根据编辑类型对任务进行分类

    Args:
        json_dir: 包含编辑提示JSON文件的目录

    Returns:
        按编辑类型分类的JSON文件路径字典
        key格式: "edit_type" 或 "main_type_sub_type" (对于混合类型)
    """
    tasks_by_type = {}

    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)

    print(f"找到 {len(json_files)} 个JSON文件")

    for json_file in tqdm(json_files, desc="分类任务"):
        data = load_json_file(json_file)
        if data is None:
            continue

        edit_type = get_edit_type(data)
        if edit_type is None:
            print(f"警告: {json_file} 中没有找到有效的edit_type")
            continue

        # 处理混合编辑类型
        if edit_type in ["hybrid", "content_memory", "content_understand", "version_backtrack"]:
            # 对于这些类型，需要根据edit_type列表中的具体操作类型来确定脚本
            edit_type_list = data.get("edit_type", [])
            if isinstance(edit_type_list, list) and len(edit_type_list) > 0:
                # 使用 "main_type_sub_type" 作为key，例如 "hybrid_add", "content_memory_replace"
                key = f"{edit_type}_{edit_type_list[0]}"
            else:
                key = edit_type
        else:
            key = edit_type

        if key not in tasks_by_type:
            tasks_by_type[key] = []
        tasks_by_type[key].append(json_file)

    return tasks_by_type


def get_script_path(edit_type: str, workflow_dir: str) -> tuple:
    """
    根据编辑类型获取对应的脚本路径和期望的JSON文件名

    Args:
        edit_type: 编辑类型（格式: "type" 或 "main_type_sub_type"）
        workflow_dir: inpaint-workflow目录路径

    Returns:
        (脚本路径, 期望的JSON文件名, task_type参数)
    """
    script_name = EDIT_TYPE_TO_SCRIPT.get(edit_type)
    expected_json_name = None
    task_type = None

    if script_name is None:
        # 处理混合类型，格式: "main_type_sub_type"
        if "_" in edit_type:
            main_type, sub_type = edit_type.split("_", 1)
            script_map = EDIT_TYPE_TO_SCRIPT.get(main_type)
            if isinstance(script_map, dict):
                script_name = script_map.get(sub_type)
                # 对于混合类型，脚本期望的JSON文件名可能是 "compose" 或 "multi-premise"
                if main_type == "hybrid":
                    expected_json_name = "compose"
                    task_type = "compose"
                elif main_type == "content_memory":
                    expected_json_name = "multi-premise"
                    task_type = "multi-premise"
                elif main_type in ["content_understand", "version_backtrack"]:
                    # 这些类型可能使用不同的命名
                    expected_json_name = main_type
                    task_type = "omit-refer" if main_type == "content_understand" else "version_refer"

        if script_name is None:
            raise ValueError(f"未找到编辑类型 {edit_type} 对应的脚本")
    else:
        # 单轮编辑类型，JSON文件名就是编辑类型
        expected_json_name = edit_type

    script_path = os.path.join(workflow_dir, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"脚本不存在: {script_path}")

    return script_path, expected_json_name, task_type


def execute_edit_tasks(
    edit_type: str,
    json_files: List[str],
    workflow_dir: str,
    base_img_dir: str,
    result_dir: str,
    use_dlc: bool = True,
):
    """
    执行特定类型的编辑任务

    Args:
        edit_type: 编辑类型
        json_files: JSON文件路径列表
        workflow_dir: inpaint-workflow目录路径
        base_img_dir: 原始图片目录
        result_dir: 结果输出目录
        use_dlc: 是否使用DLC分布式执行（默认True）
    """
    if len(json_files) == 0:
        print(f"编辑类型 {edit_type} 没有任务需要处理")
        return

    print(f"\n处理编辑类型: {edit_type}, 任务数量: {len(json_files)}")

    # 获取脚本路径和期望的JSON文件名
    try:
        script_path, expected_json_name, task_type = get_script_path(edit_type, workflow_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"错误: {e}")
        return

    # 对于混合编辑类型，可能需要创建符号链接以匹配脚本期望的格式
    # 步骤4生成的JSON文件名是 {instruction_type}.json (如 hybrid.json)
    # 但脚本可能期望不同的文件名 (如 compose.json)
    created_links = []
    if expected_json_name:
        actual_json_name = os.path.basename(json_files[0]).replace(".json", "")
        if expected_json_name != actual_json_name:
            # 需要为每个JSON文件创建符号链接
            print(f"创建符号链接: {actual_json_name}.json -> {expected_json_name}.json")
            for json_file in json_files:
                json_dir = os.path.dirname(json_file)
                link_path = os.path.join(json_dir, f"{expected_json_name}.json")
                if not os.path.exists(link_path):
                    try:
                        os.symlink(os.path.basename(json_file), link_path)
                        created_links.append(link_path)
                    except OSError as e:
                        # 如果符号链接失败，尝试复制文件
                        import shutil
                        shutil.copy2(json_file, link_path)
                        created_links.append(link_path)
                        print(f"  使用文件复制代替符号链接: {link_path}")

    # 提取所有JSON文件所在的目录（去重）
    # 注意：inpaint-workflow脚本期望JSON文件按照 {json_dir}/**/{expected_json_name}.json 的格式组织
    json_dirs = set()
    for json_file in json_files:
        json_dirs.add(os.path.dirname(json_file))

    # 使用包含所有文件的父目录作为json_dir
    # 脚本会使用glob模式查找所有匹配的JSON文件
    if len(json_dirs) == 1:
        json_dir = list(json_dirs)[0]
    else:
        # 找到所有JSON文件的公共父目录
        json_dir = os.path.commonpath([os.path.dirname(f) for f in json_files])

    # 构建命令参数
    cmd_args = [
        "--json_dir", json_dir,
        "--base_img_dir", base_img_dir,
        "--result_dir", result_dir,
    ]

    # 添加task_type参数（如果需要）
    if task_type:
        cmd_args.extend(["--task_type", task_type])

    # 构建完整命令
    cmd = [sys.executable, script_path] + cmd_args

    print(f"执行命令: {' '.join(cmd)}")
    print(f"JSON目录: {json_dir}")
    print(f"结果目录: {result_dir}")

    try:
        # 切换到workflow目录执行
        # 注意：这些脚本使用dlc_context_runner，需要设置相应的环境变量
        # 如果使用DLC，需要确保以下环境变量已设置：
        # - RANK, MASTER_ADDR, MASTER_PORT, WORLD_SIZE
        # - PYTHONPATH, COMFYUI_PATH
        result = subprocess.run(
            cmd,
            cwd=workflow_dir,
            check=True,
            capture_output=False,
        )
        print(f"✓ 编辑类型 {edit_type} 处理完成")
    except subprocess.CalledProcessError as e:
        print(f"✗ 执行失败: {e}")
        print(f"提示: 请确保已设置必要的环境变量（RANK, MASTER_ADDR, MASTER_PORT等）")
        raise
    finally:
        # 清理创建的符号链接
        for link_path in created_links:
            try:
                if os.path.islink(link_path):
                    os.unlink(link_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="根据编辑类型执行图片编辑任务"
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        help="包含编辑提示JSON文件的目录（步骤4的输出）"
    )
    parser.add_argument(
        "--base_img_dir",
        type=str,
        required=True,
        help="原始图片目录"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="编辑结果输出目录"
    )
    parser.add_argument(
        "--workflow_dir",
        type=str,
        default=None,
        help="inpaint-workflow目录路径（默认：项目根目录下的inpaint-workflow）"
    )
    parser.add_argument(
        "--edit_types",
        type=str,
        nargs="+",
        default=None,
        help="指定要处理的编辑类型（默认：处理所有类型）"
    )
    parser.add_argument(
        "--no-dlc",
        action="store_true",
        help="不使用DLC分布式执行（用于测试）"
    )

    args = parser.parse_args()

    # 确定workflow目录
    if args.workflow_dir is None:
        # 默认使用项目根目录下的inpaint-workflow
        project_root = Path(__file__).parent.parent
        args.workflow_dir = str(project_root / "inpaint-workflow")

    if not os.path.exists(args.workflow_dir):
        raise FileNotFoundError(f"workflow目录不存在: {args.workflow_dir}")

    # 分类任务
    print("正在分类任务...")
    tasks_by_type = classify_tasks_by_type(args.json_dir)

    if len(tasks_by_type) == 0:
        print("没有找到需要处理的任务")
        return

    print(f"\n找到以下编辑类型: {list(tasks_by_type.keys())}")
    print(f"各类型任务数量:")
    for edit_type, files in tasks_by_type.items():
        print(f"  {edit_type}: {len(files)}")

    # 过滤编辑类型
    if args.edit_types:
        tasks_by_type = {
            k: v for k, v in tasks_by_type.items()
            if k in args.edit_types or any(k.startswith(et) for et in args.edit_types)
        }
        print(f"\n过滤后需要处理的编辑类型: {list(tasks_by_type.keys())}")

    # 按顺序处理每种编辑类型
    # 单轮编辑类型优先处理
    single_round_types = ["add", "remove", "replace", "adjust", "background", "extract"]
    multi_round_types = [k for k in tasks_by_type.keys() if k not in single_round_types]

    all_types = [t for t in single_round_types if t in tasks_by_type] + multi_round_types

    for edit_type in all_types:
        if edit_type not in tasks_by_type:
            continue

        try:
            execute_edit_tasks(
                edit_type=edit_type,
                json_files=tasks_by_type[edit_type],
                workflow_dir=args.workflow_dir,
                base_img_dir=args.base_img_dir,
                result_dir=args.result_dir,
                use_dlc=not args.no_dlc,
            )
        except Exception as e:
            print(f"处理编辑类型 {edit_type} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n所有编辑任务处理完成！")


if __name__ == "__main__":
    main()

