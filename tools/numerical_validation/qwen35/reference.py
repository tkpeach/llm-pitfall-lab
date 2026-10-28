#!/usr/bin/env python3
"""Run the fixed MLX-LM reference in its own process; never imported by the C++ worker."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import time


def main() -> None:
    import mlx.core as mx
    from mlx_lm.utils import load_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=0)
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(10 * 1024**3)
    mx.reset_peak_memory()
    started = time.monotonic()
    model, config = load_model(args.artifact)
    model.eval()
    tokens = json.loads(args.input.read_text())["tokens"]
    ids = mx.array([tokens], mx.int32)
    cache = model.make_cache()
    observations = {"embedding": model.model.embed_tokens(ids)}
    manual_first_layer = None
    if os.environ.get("LAB_D4_DIAGNOSTIC") and args.chunk == 0:
        import mlx.nn as nn
        from mlx_lm.models.gated_delta import normalize_qk, compute_g, gated_delta_update
        from mlx_lm.models.activations import swiglu
        layer = model.layers[0]
        d = layer.linear_attn
        x = observations["embedding"]
        n = layer.input_layernorm(x)
        qkv = d.in_proj_qkv(n)
        joined = mx.concatenate([mx.zeros((1, d.conv_kernel_size-1, d.conv_dim), n.dtype), qkv], axis=1)
        conv_pre = d.conv1d(joined)
        conv = nn.silu(conv_pre)
        q, k, v = [t.reshape(1, len(tokens), heads, dim) for t, heads, dim in zip(
            mx.split(conv, [d.key_dim, 2*d.key_dim], -1),
            [d.num_k_heads, d.num_k_heads, d.num_v_heads], [d.head_k_dim, d.head_k_dim, d.head_v_dim])]
        q, k = normalize_qk(q, k, inv_scale=d.head_k_dim**-0.5, eps=1e-6)
        a, b = d.in_proj_a(n), d.in_proj_b(n)
        decay, beta = compute_g(d.A_log, a, d.dt_bias), mx.sigmoid(b)
        raw, _ = gated_delta_update(q, k, v, a, b, d.A_log, d.dt_bias, use_kernel=True)
        z = d.in_proj_z(n).reshape(1, len(tokens), d.num_v_heads, d.head_v_dim)
        normalized = mx.fast.rms_norm(raw, d.norm.weight, d.norm.eps)
        gated = d.norm(raw, z)
        mixed = d.out_proj(gated.reshape(1, len(tokens), -1))
        h = x + mixed
        post = layer.post_attention_layernorm(h)
        gate, up = layer.mlp.gate_proj(post), layer.mlp.up_proj(post)
        glu = swiglu(gate, up)
        mlp_out = layer.mlp.down_proj(glu)
        manual_first_layer = h + mlp_out
        for name, value in {"inputnorm": n, "postnorm": post, "mlp.gate": gate, "mlp.up": up,
                            "mlp.glu": glu, "mlp.out": mlp_out}.items():
            observations[f"layer.0.{name}"] = value
        for name, value in {"qkv": qkv, "conv_pre": conv_pre, "conv": conv, "q": q, "k": k,
                            "v": v, "decay": decay, "beta": beta, "raw_out": raw, "z": z,
                            "normalized": normalized, "gated": gated, "out": mixed}.items():
            observations[f"layer.0.delta.{name}"] = value
    layer_type = type(model.layers[0])
    original_call = layer_type.__call__
    indices = {id(layer): i for i, layer in enumerate(model.layers)}

    def observed_call(layer, *args, **kwargs):
        output = original_call(layer, *args, **kwargs)
        index = indices[id(layer)]
        observations[f"layer.{index}"] = output
        return output

    layer_type.__call__ = observed_call
    forward_start = time.monotonic()
    try:
        if args.chunk == 0:
            hidden = model.model(ids, cache=cache)
            logits = model.model.embed_tokens.as_linear(hidden)
            observations["final_hidden"] = hidden
        else:
            parts = []
            for begin in range(0, len(tokens), args.chunk):
                parts.append(model(ids[:, begin:begin+args.chunk], cache=cache))
                pending = [parts[-1]]
                for i, c in enumerate(cache):
                    pending.extend(c.keys_and_values() if (i+1)%4 == 0 else [c[0], c[1]])
                mx.eval(pending)
            logits = mx.concatenate(parts, axis=1)
            observations = {}
    finally:
        layer_type.__call__ = original_call
    if manual_first_layer is not None and not mx.array_equal(manual_first_layer, observations["layer.0"]).item():
        raise AssertionError("diagnostic decomposition differs from the actual reference layer")
    observations["logits"] = logits
    for i, c in enumerate(cache):
        if (i + 1) % 4 == 0:
            observations[f"cache.layer.{i}.keys"], observations[f"cache.layer.{i}.values"] = c.keys_and_values()
        else:
            observations[f"cache.layer.{i}.conv"] = c[0]
            observations[f"cache.layer.{i}.recurrent"] = c[1]
    mx.eval(list(observations.values()))
    prefill_seconds = time.monotonic() - forward_start
    generated = []
    next_logits = logits[:, -1, :]
    for _ in range(8):
        token = mx.argmax(next_logits, -1).item()
        generated.append(token)
        if token in (248044, 248046):
            break
        next_logits = model(mx.array([[token]], mx.int32), cache=cache)[:, -1, :]
        mx.eval(next_logits)
    observations["generated"] = mx.array(generated, mx.int32)
    mx.save_safetensors(str(args.output), observations)
    print(json.dumps({"reference": "mlx-lm==0.32.2", "gdn_packed": os.environ.get("MLX_GDN_PACKED", "1"), "chunk": args.chunk, "input_tokens": len(tokens),
                      "prefill_seconds": prefill_seconds, "total_seconds": time.monotonic() - started,
                      "mlx_peak_bytes": mx.get_peak_memory(),
                      "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "generated": generated}), flush=True)


if __name__ == "__main__":
    main()
