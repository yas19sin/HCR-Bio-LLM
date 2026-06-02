from __future__ import annotations

import argparse

import torch

from src.eval.generation import generate_text
from src.training.trainer import load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="First ")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device({"device": args.device})
    model, tokenizer, _, meta = load_checkpoint(args.checkpoint, device)
    with torch.no_grad():
        text = generate_text(
            model,
            tokenizer,
            args.prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(f"# checkpoint step: {meta['step']}")
    print(text)


if __name__ == "__main__":
    main()

