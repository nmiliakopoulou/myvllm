import json
import time
import threading
import subprocess
import requests
import numpy as np
import pandas as pd
from typing import List, Tuple
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

# Constants
TRACE_FILE = "/home/nmi4/vllm/benchmarks/my_benchmarks/azure_traces.csv"
METRICS_FILE = "/home/nmi4/vllm/benchmarks/my_benchmarks/gpu_metrics.log"
RANDOM_SEED = 42
OVERSAMPLING_FACTOR = 2.0

# Load Azure trace dataset
def sample_requests_from_csv(trace_file: str) -> List[Tuple[float, int, int]]:
    """Loads requests from an Azure trace CSV file."""
    df = pd.read_csv(trace_file, header=None, names=["timestamp", "input_len", "output_len"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_time = df["timestamp"].min()
    df["elapsed_time"] = (df["timestamp"] - start_time).dt.total_seconds()

    return list(df.itertuples(index=False, name=None))

# Poisson Load Generation with Activity Window Enforcement
def EnforceActivityWindow(start_time, end_time, instance_events):
    """Ensures event timestamps stay within a specified time window."""
    events_abs = [0] + instance_events
    event_times = []
    last_value = 0
    for prevInd in range(1, len(events_abs) + 1):
        last_value += events_abs[prevInd - 1]
        event_times.append(last_value)

    event_times = [e for e in event_times if start_time < e < end_time]
    return [event_times[0]] + [event_times[i] - event_times[i - 1] for i in range(1, len(event_times))]

def generate_poisson_distribution(rates, duration):
    """Generates Poisson-distributed request arrival times within a time window."""
    np.random.seed(RANDOM_SEED)
    ret = []
    for rate in rates:
        beta = 1.0 / rate
        inter_arrivals = list(np.random.exponential(scale=beta, size=int(OVERSAMPLING_FACTOR * duration * rate)))
        ret.append(EnforceActivityWindow(0, duration, inter_arrivals))
    return ret

# Request Sending Function
def send_request(backend: str, model: str, api_url: str, input_len: int, output_len: int) -> None:
    """Sends a request to the server and records latency."""
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    pload = {
        "prompt": "a" * (input_len//4),
        "n": 1,
        "best_of": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": output_len//4,
        "ignore_eos": True,
        "stream": False,
    }
    if model:
        pload["model"] = model

    response = requests.post(api_url, headers=headers, json=pload)
    if response.status_code != 200:
        print("ERROR:", response.text)

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time

# Poisson-based request generator
def generate_poisson_load(load_reqs: int, duration: int):
    """Generates Poisson-distributed request load while preserving input/output lengths from trace."""
    dataset = sample_requests_from_csv(TRACE_FILE)
    dataset_size = len(dataset)
    

    instance_events = generate_poisson_distribution([load_reqs], duration)[0]

    after_time, before_time = 0, 0
    st = 0
    threads = []

    i=0
    print(len(instance_events))
    for t in instance_events:
        st = st + t - (after_time - before_time)
        before_time = time.time()
        if st > 0:
            time.sleep(st)

        #index = np.random.randint(0, dataset_size)
        _, input_len, output_len, _ = dataset[i%dataset_size]
        i+=1

        thread = threading.Thread(target=send_request, args=("vllm", None, "http://localhost:8000/generate", input_len, output_len))
        thread.start()
        threads.append(thread)

        after_time = time.time()

    for thread in threads:
        thread.join()
        print("hi")

# Monitoring Functions
def start_process_dcgmi():
    """Starts DCGMI monitoring process."""
    command = "dcgmi dmon -i 0 -e 100,101,112,156,157,140,150,203,204,1002,1003 -d 1000 > dcgm_monitor_test"
    return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def check_process_dcgmi(process):
    """Checks if DCGMI monitoring process is running."""
    return process.poll() is None

def restart_process_dcgmi(process):
    """Restarts DCGMI monitoring process if it stops."""
    process.kill()
    return start_process_dcgmi()

def check_dcgmi():
    """Continuously checks and restarts DCGMI if needed."""
    process = start_process_dcgmi()
    while True:
        time.sleep(20)
        if not check_process_dcgmi(process):
            process = restart_process_dcgmi(process)

def export_metrics():
    """Exports GPU and memory utilization metrics from DCGMI logs and writes them to a file."""
    readfile = "dcgm_monitor_test"
    with open(METRICS_FILE, "a") as f:
        while True:
            time.sleep(1)
            result = subprocess.run(["tail", "-n", "1", readfile], stdout=subprocess.PIPE)
            last_line = result.stdout.decode("utf-8").strip()
            if not last_line:
                continue
            try:
                metrics = last_line.split()
                power = float(metrics[6])  # Example: Power usage
            except:
                power = 120.0  # Default value if parsing fails
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"{timestamp}, Power: {power}W\n")
            f.flush()  # Ensure the log is updated in real-time

if __name__ == "__main__":
    # Start monitoring threads
    thread_dcgmi = threading.Thread(target=check_dcgmi)
    thread_dcgmi.start()

    thread_export_metrics = threading.Thread(target=export_metrics)
    thread_export_metrics.start()

    # Choose Poisson-based request load
    rps = 5  # Requests per second
    duration = 600  # Run for 900 seconds (15 minutes)
    thread_load = threading.Thread(target=generate_poisson_load, args=(rps, duration))

    thread_load.start()
    thread_load.join()
    print("hi2")
    thread_export_metrics.join()
    print("ok")
    thread_dcgmi.join()
    print("end")
