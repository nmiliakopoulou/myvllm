"""Benchmark the latency of processing a single batch of requests."""
import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import List, Optional
import os
import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser


def main(args: argparse.Namespace):
    print(args)

    engine_args = EngineArgs.from_cli_args(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: List[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def llm_generate():
        if not args.use_beam_search:
            llm.generate(dummy_prompts,
                         sampling_params=sampling_params,
                         use_tqdm=False)
        else:
            llm.beam_search(
                dummy_prompts,
                BeamSearchParams(
                    beam_width=args.n,
                    max_tokens=args.output_len,
                    ignore_eos=True,
                ))

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir))) as p:
                llm_generate()
            print(p.key_averages().table(sort_by="self_cuda_time_total"))
        else:
            start_time = time.perf_counter()
            llm_generate()
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    with open('/home/nmi4/moe_exp_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    with open('/home/nmi4/moe_gate_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    with open('/home/nmi4/moe_shared_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    with open('/home/nmi4/inference_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    with open('/home/nmi4/attn_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    with open('/home/nmi4/moe_total_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    with open('/home/nmi4/moe_layernorm_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    with open('/home/nmi4/norm_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    with open('/home/nmi4/layer_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    with open('/home/nmi4/inpemb_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    with open('/home/nmi4/inpemp2_latency.txt', 'w') as myfile:
        print("NOW IT'S WARMING TIME", file = myfile, flush=True)
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)
    
    with open('/home/nmi4/moe_exp_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)
    with open('/home/nmi4/moe_gate_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)
    with open('/home/nmi4/moe_shared_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)
    with open('/home/nmi4/attn_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)
    with open('/home/nmi4/moe_total_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)
    with open('/home/nmi4/moe_layernorm_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)
    with open('/home/nmi4/inpemb_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)
    with open('/home/nmi4/inpemp2_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)
    with open('/home/nmi4/layer_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)
    with open('/home/nmi4/norm_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)
    with open('/home/nmi4/inference_latency.txt', 'a') as myfile:
        print("NOW IT'S ACTUAL ITERATION TIME", file = myfile)

    if args.profile:
        profile_dir = args.profile_result_dir
        if not profile_dir:
            profile_dir = Path(
                "."
            ) / "vllm_benchmark_result" / f"latency_result_{time.time()}"
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f'Avg latency: {np.mean(latencies)} seconds')
    for percentage, percentile in zip(percentages, percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')

    os.rename("/home/nmi4/moe_exp_latency.txt", f"/home/nmi4/moe_exp_latency_{args.input_len}_{args.batch_size}.txt")
    os.rename("/home/nmi4/moe_gate_latency.txt", f"/home/nmi4/moe_gate_latency_{args.input_len}_{args.batch_size}.txt")
    os.rename("/home/nmi4/moe_shared_latency.txt", f"/home/nmi4/moe_shared_latency_{args.input_len}_{args.batch_size}.txt")
    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--input-len', type=int, default=64)
    parser.add_argument('--output-len', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the latency results in JSON format.')

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)


# TTFT Qwen-1.5-MoE-A2.7B 280ms (512), 170ms (256), 120ms (128), 99ms (64)
# TTFT Qwen-1.5-MoE-A2.7B-Chat 280ms (512), 167ms (256), 120ms (128), 99ms (64)
# TBT Qwen-1.5-MoE-A2.7B-Chat IS:64, OS:128 P50 26.7ms -- P99 28.1ms 

# TTFT Qwen1.5-14B-Chat 1020ms (512), 541ms (256), 300ms (128), 182ms (64)
# TBT Qwen1.5-14B-Chat IS:64, OS:128 P50 53.1ms -- P99 53.1ms


# TTFT Qwen-1.5-MoE-A2.7B-Chat 525ms (512, 16), 1006ms (512, 32)
# TTFT Qwen-1.5-7B-Chat  104ms (64)