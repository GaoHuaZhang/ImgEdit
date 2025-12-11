import os
import requests
import time
import threading
from flask import Flask, request

master_addr = os.environ.get("MASTER_ADDR")
master_port = os.environ.get("MASTER_PORT")
world_size = int(os.environ.get("WORLD_SIZE"))

workers: list[str] = []
app = Flask(__name__)


@app.route("/")
def ping():
    worker_addr = request.args.get("worker_addr")
    print(f"[Master] Worker {worker_addr} is registering")
    workers.append(worker_addr)
    return "pong"


def run_flask():
    app.run(host=master_addr, port=master_port)


def dlc_context_runner(func, wait_time=120, *args, **kwargs):
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # 单机模式：如果 WORLD_SIZE=1，直接使用本地 worker
    if world_size == 1:
        print(f"[Master] Single node mode (WORLD_SIZE=1), using local worker")
        # 使用本地地址和默认端口
        local_ip = os.environ.get("MASTER_ADDR", "localhost")
        if local_ip == "localhost" or local_ip == "127.0.0.1":
            # 尝试获取真实 IP
            try:
                from imgedit.test_dlc.ip_utils import get_local_ip
                local_ip = get_local_ip() or "localhost"
            except:
                local_ip = "localhost"

        # 检查本地是否有 ComfyUI 服务运行
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        local_workers = [f"{local_ip}:{8180 + i}" for i in range(gpu_count)]

        # 验证 worker 是否可用
        print(f"[Master] Checking local workers: {local_workers}")
        for worker in local_workers:
            for _ in range(wait_time):
                try:
                    resp = requests.post(f"http://{worker}/prompt", timeout=2)
                    print(f"[Master] Worker {worker} is ready")
                    break
                except Exception as e:
                    time.sleep(1)
                    if _ == wait_time - 1:
                        print(f"[Master] Warning: Worker {worker} not available, continuing anyway")

        print(f"[Master] Using local workers: {local_workers}")
        func(*args, **kwargs, workers=local_workers)
        return

    # 多节点模式：等待 worker 注册
    while len(workers) < world_size - 1:
        time.sleep(1)
        print(f"[Master] Waiting for {len(workers)}/{world_size - 1} workers to register")

    print(f"[Master] All {world_size - 1} workers registered")

    # 验证 worker 可用性
    for _ in range(wait_time):
        all_ready = True
        try:
            for worker in workers:
                print(f"[Master] Checking worker {worker}")
                if worker is None or "None" in str(worker):
                    print(f"[Master] Warning: Invalid worker address {worker}, skipping")
                    all_ready = False
                    continue
                resp = requests.post(f"http://{worker}/prompt", timeout=5)
                print(f"[Master] Worker {worker} is ready")
        except Exception as e:
            print(f"[Master] Error checking worker: {e}")
            all_ready = False
            time.sleep(1)
            continue

        if all_ready:
            break
    else:
        print(f"[Master] Warning: Some workers may not be ready, continuing anyway")

    print(f"[Master] All workers are ready")

    func(*args, **kwargs, workers=workers)
