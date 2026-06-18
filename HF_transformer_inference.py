from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hugging Face causal-LM inference for Nsight profiling. "
            "Supports isolated prefill, decode, or both phases."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="Hugging Face model id (default: distilgpt2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device. Defaults to cuda if available.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts per batch.")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["generate", "prefill", "decode", "both"],
        default="both",
        help=(
            "generate: model.generate() (prefill+decode fused). "
            "prefill: one parallel forward over --input-tokens. "
            "decode: profile only single-token decode steps (warmup prefill outside capture). "
            "both: prefill then decode in one iteration with separate NVTX ranges."
        ),
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=64,
        help=(
            "Target prompt length in tokens for prefill/decode/both modes. "
            "Longer prompts = more prefill work (processed in parallel)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Decode steps for generate/both/decode modes (one forward per new token).",
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog",
        help="Seed text; padded or truncated to --input-tokens.",
    )
    parser.add_argument(
        "--use-cuda-profiler-api",
        action="store_true",
        help="Wrap profiled loop with cudaProfilerStart/Stop (for ncu).",
    )
    parser.add_argument("--sleep-before-profile-s", type=float, default=0.0)
    return parser.parse_args()


def make_batch_prompts(base: str, batch_size: int) -> List[str]:
    return [f"{base} [sample {i}]" for i in range(batch_size)]


def build_fixed_length_batch(
    tokenizer,
    prompts: List[str],
    num_input_tokens: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompts and pad/truncate each row to exactly num_input_tokens."""
    pad_id = tokenizer.pad_token_id
    rows: List[List[int]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) < num_input_tokens:
            ids = ids + [pad_id] * (num_input_tokens - len(ids))
        else:
            ids = ids[:num_input_tokens]
        rows.append(ids)

    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = (input_ids != pad_id).to(dtype=torch.long, device=device)
    # Left padding would be more realistic for batched decode, but right-pad is fine for profiling.
    return input_ids, attention_mask


@torch.inference_mode()
def run_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    nvtx_label: str = "prefill",
) -> torch.Tensor:
    torch.cuda.nvtx.range_push(nvtx_label)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    torch.cuda.nvtx.range_pop()
    return outputs


@torch.inference_mode()
def run_decode_steps(
    model: torch.nn.Module,
    prefill_outputs,
    decode_steps: int,
    nvtx_outer: str = "decode",
    nvtx_per_step: bool = False,
) -> None:
    past_key_values = prefill_outputs.past_key_values
    next_token = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.nvtx.range_push(nvtx_outer)
    for step in range(decode_steps):
        label = f"decode_step_{step}" if nvtx_per_step else nvtx_outer
        if nvtx_per_step:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(label)

        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    if nvtx_per_step and decode_steps > 0:
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()


@torch.inference_mode()
def run_prefill_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decode_steps: int,
    nvtx_per_decode_step: bool = True,
) -> None:
    prefill_out = run_prefill(model, input_ids, attention_mask, nvtx_label="phase_prefill")
    run_decode_steps(
        model,
        prefill_out,
        decode_steps,
        nvtx_outer="phase_decode",
        nvtx_per_step=nvtx_per_decode_step,
    )


@torch.inference_mode()
def run_generation(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: torch.device,
    num_input_tokens: Optional[int] = None,
) -> None:
    if num_input_tokens is not None:
        input_ids, attention_mask = build_fixed_length_batch(
            tokenizer, prompts, num_input_tokens, device
        )
    else:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

    torch.cuda.nvtx.range_push("hf_generate")
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.cuda.nvtx.range_pop()


def run_iteration(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    phase: str,
    input_tokens: int,
    decode_steps: int,
) -> None:
    input_ids, attention_mask = build_fixed_length_batch(
        tokenizer, prompts, input_tokens, device
    )

    if phase == "prefill":
        run_prefill(model, input_ids, attention_mask, nvtx_label="phase_prefill")
    elif phase == "decode":
        prefill_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        run_decode_steps(
            model,
            prefill_out,
            decode_steps,
            nvtx_outer="phase_decode",
            nvtx_per_step=True,
        )
    elif phase == "both":
        run_prefill_decode(model, input_ids, attention_mask, decode_steps)
    elif phase == "generate":
        run_generation(
            model,
            tokenizer,
            prompts,
            decode_steps,
            device,
            num_input_tokens=input_tokens,
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")


def run_warmup_iteration(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    phase: str,
    input_tokens: int,
    decode_steps: int,
) -> None:
    # For decode-only profiling, prefill still happens each iter but outside NVTX labels.
    run_iteration(model, tokenizer, prompts, device, phase, input_tokens, decode_steps)


def cuda_profiler_start() -> None:
    if torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop() -> None:
    if torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError("Nsight GPU profiling expects CUDA.")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    prompts = make_batch_prompts(args.prompt, args.batch_size)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Phase: {args.phase}")
    print(f"Batch size: {args.batch_size}")
    print(f"Input tokens (prefill length): {args.input_tokens}")
    print(f"Decode steps: {args.max_new_tokens}")
    print(f"dtype: {dtype}")
    print(f"Warmup iters: {args.warmup_iters} | Profile iters: {args.profile_iters}")

    if args.phase == "both":
        print(
            "Experiment: compare NVTX ranges 'phase_prefill' vs 'phase_decode' in Nsight. "
            "Increase --input-tokens for heavier prefill; increase --max-new-tokens for heavier decode."
        )

    torch.cuda.nvtx.range_push("warmup")
    for _ in range(args.warmup_iters):
        run_warmup_iteration(
            model,
            tokenizer,
            prompts,
            device,
            args.phase,
            args.input_tokens,
            args.max_new_tokens,
        )
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print("Warmup complete.")

    if args.sleep_before_profile_s > 0:
        time.sleep(args.sleep_before_profile_s)

    profiler_ctx = nullcontext()
    if args.use_cuda_profiler_api:
        profiler_ctx = _CudaProfilerContext()

    prefill_ms: List[float] = []
    decode_ms: List[float] = []

    torch.cuda.nvtx.range_push("profiled_inference_loop")
    with profiler_ctx:
        t0 = time.perf_counter()
        for i in range(args.profile_iters):
            torch.cuda.nvtx.range_push(f"inference_iter_{i}")

            if args.phase == "both":
                input_ids, attention_mask = build_fixed_length_batch(
                    tokenizer, prompts, args.input_tokens, device
                )
                torch.cuda.synchronize()
                t_prefill = time.perf_counter()
                prefill_out = run_prefill(
                    model, input_ids, attention_mask, nvtx_label="phase_prefill"
                )
                torch.cuda.synchronize()
                prefill_ms.append((time.perf_counter() - t_prefill) * 1000)

                t_decode = time.perf_counter()
                run_decode_steps(
                    model,
                    prefill_out,
                    args.max_new_tokens,
                    nvtx_outer="phase_decode",
                    nvtx_per_step=True,
                )
                torch.cuda.synchronize()
                decode_ms.append((time.perf_counter() - t_decode) * 1000)
            else:
                run_iteration(
                    model,
                    tokenizer,
                    prompts,
                    device,
                    args.phase,
                    args.input_tokens,
                    args.max_new_tokens,
                )
                torch.cuda.synchronize()

            torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    torch.cuda.nvtx.range_pop()

    print(f"Profiled loop: {elapsed:.3f}s")
    if prefill_ms and decode_ms:
        avg_prefill = sum(prefill_ms) / len(prefill_ms)
        avg_decode = sum(decode_ms) / len(decode_ms)
        per_decode = avg_decode / max(args.max_new_tokens, 1)
        print(f"Avg prefill ({args.input_tokens} tokens): {avg_prefill:.2f} ms")
        print(f"Avg decode ({args.max_new_tokens} steps): {avg_decode:.2f} ms")
        print(f"Avg per decode step: {per_decode:.2f} ms")
        print(f"Prefill/decode time ratio: {avg_prefill / max(avg_decode, 1e-6):.2f}x")
    print("Done. In Nsight, filter NVTX for phase_prefill and phase_decode.")


class _CudaProfilerContext:
    def __enter__(self) -> None:
        cuda_profiler_start()

    def __exit__(self, exc_type, exc, tb) -> None:
        cuda_profiler_stop()


if __name__ == "__main__":
    main()
