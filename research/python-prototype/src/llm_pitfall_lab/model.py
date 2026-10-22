"""Fixed Qwen3.5-4B artifact preparation and bounded text inference."""

import hashlib
import json
import os
import time
import tomllib
from pathlib import Path

from .checkpoint import canonical
from .runner import PauseControl, worker_lock


def _configuration(root):
    with (root / "configs/model-4b.toml").open("rb") as handle:
        config = tomllib.load(handle)
    # Keep all caches inside the budgeted project, before importing Hub tooling.
    os.environ["HF_HOME"] = str(root / ".cache/huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    destination = (root / config["local_directory"]).resolve()
    if not destination.is_relative_to(root / "models"):
        raise ValueError("model directory must be under models/")
    return config, destination


def prepare(root: Path, guard, download: bool) -> dict:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    config, destination = _configuration(root)
    guard.check(1024**2)
    info = HfApi().model_info(config["execution_repo"], revision=config["execution_revision"],
                             files_metadata=True, timeout=30, token=False)
    if info.sha != config["execution_revision"]:
        raise ValueError("resolved revision differs from fixed execution identity")
    small = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "generation_config.json", "chat_template.jinja", "model.safetensors.index.json",
             "tokenizer.model", "vocab.json", "merges.txt"}
    files = [f for f in info.siblings if f.rfilename in small or f.rfilename.endswith(".safetensors")]
    if not files or any(f.size is None for f in files):
        raise ValueError("incomplete file-size metadata")
    names = {f.rfilename for f in files}
    if not {"config.json", "tokenizer_config.json"}.issubset(names):
        raise ValueError("missing model/tokenizer configuration")
    total = sum(f.size for f in files)
    # Account conservatively for staging/copies and a small manifest reserve.
    disk = guard.check(2 * total + 128 * 1024**2)
    plan = {"config": config, "files": [{"path": f.rfilename, "bytes": f.size} for f in files],
            "download_bytes": total, "reserved_peak_additional_bytes": 2 * total + 128 * 1024**2,
            "preflight": disk, "status": "planned"}
    if not download:
        return plan
    destination.mkdir(parents=True, exist_ok=True)
    # Validate the small config before downloading weights.
    path = hf_hub_download(config["execution_repo"], "config.json", revision=info.sha,
                           local_dir=destination, token=False)
    model_config = json.loads(Path(path).read_text())
    quantization = model_config.get("quantization", {})
    if model_config.get("model_type") != "qwen3_5" or quantization.get("bits") != 4:
        raise ValueError("execution pack is not the selected Qwen3.5 4-bit model")
    snapshot_download(config["execution_repo"], revision=info.sha, local_dir=destination,
                      allow_patterns=sorted(names), token=False, max_workers=1)
    hashes = {}
    for file in files:
        local = destination / file.rfilename
        if local.stat().st_size != file.size:
            raise ValueError(f"download size mismatch: {file.rfilename}")
        with local.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if file.lfs and file.lfs.sha256 != digest:
            raise ValueError(f"upstream content hash mismatch: {file.rfilename}")
        hashes[file.rfilename] = digest
    plan.update(status="downloaded_verified", sha256=hashes,
                quantization=quantization, note="file verification only; not model acceptance")
    guard.check(len(canonical(plan)) + 65536)
    (destination / "artifact-manifest.json").write_bytes(canonical(plan))
    return plan


def infer(root: Path, run_dir: Path, guard, case_id: str, max_tokens: int) -> dict:
    config, destination = _configuration(root)
    if not 1 <= max_tokens <= 512:
        raise ValueError("initial inference permits 1..512 output tokens")
    guard.check(4 * 1024**2)
    manifest = json.loads((destination / "artifact-manifest.json").read_text())
    if manifest["config"] != config or manifest["status"] != "downloaded_verified":
        raise ValueError("artifact manifest does not match fixed configuration")
    # The model runs offline; missing local files must never trigger remote fallback.
    os.environ["HF_HUB_OFFLINE"] = "1"
    from safetensors import safe_open
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from .task import score_answer

    if not mx.metal.is_available():
        raise RuntimeError("Metal required; no CPU fallback")
    mx.set_default_device(mx.gpu)
    cases = json.loads((root / "fixtures/task_contract.json").read_text())
    case = next((c for c in cases if c["id"] == case_id), None)
    if case is None:
        raise ValueError("unknown contract fixture ID")
    with worker_lock(run_dir), PauseControl(run_dir) as control:
        tensor_summary = {"total": 0, "vision": 0, "mtp": 0}
        for path in destination.glob("*.safetensors"):
            with safe_open(path, framework="numpy") as handle:
                for key in handle.keys():
                    tensor_summary["total"] += 1
                    tensor_summary["vision"] += int("visual" in key or "vision_tower" in key)
                    tensor_summary["mtp"] += int("mtp." in key)
        mx.reset_peak_memory()
        start = time.perf_counter()
        model, tokenizer = load(str(destination), trust_remote_code=False)
        mx.synchronize()
        load_seconds = time.perf_counter() - start
        system = ('从订单描述抽取唯一 JSON 对象，恰好包含 order_id、product、quantity、'
                  'unit_price_minor、currency、delivery_date。缺失为 null。数量为整数；'
                  '只提取明确单价，金额乘100转为整数；币种仅明确时填 CNY/USD。'
                  '只转换绝对日期为 YYYY-MM-DD。不从总价推断单价，不推断相对日期。'
                  '不输出解释或代码围栏。')
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": case["input"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        cache = make_prompt_cache(model)
        start = time.perf_counter()
        text, generated, response, paused = "", [], None, False
        for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens,
                                        sampler=make_sampler(temp=0), prompt_cache=cache):
            text += response.text
            generated.append(response.token)
            if control.requested():
                paused = True
                break
        mx.synchronize()
        cache_summary = []
        for index, layer_cache in enumerate(cache):
            from mlx.utils import tree_flatten
            arrays = [array for _, array in tree_flatten(layer_cache.state) if isinstance(array, mx.array)]
            cache_summary.append({"layer": index, "type": type(layer_cache).__name__,
                                  "shapes": [list(array.shape) for array in arrays],
                                  "data_bytes": sum(array.nbytes for array in arrays)})
        report = {"scope": "single cold-start fixture; not accuracy or throughput benchmark",
                  "status": "paused_partial" if paused else "completed", "artifact": config,
                  "input_tensor_summary": tensor_summary, "load_seconds": load_seconds,
                  "generation_wall_seconds": time.perf_counter() - start,
                  "case": case, "prompt_token_ids": prompt, "output": text,
                  "generation_events": len(generated), "thinking": False,
                  "mlx_peak_allocated_bytes": mx.get_peak_memory(),
                  "cache": cache_summary, "score": score_answer(text, case["gold"]),
                  "prompt_tps_library": response.prompt_tps if response else None,
                  "generation_tps_library": response.generation_tps if response else None,
                  "finish_reason": response.finish_reason if response else None}
        guard.check(len(canonical(report)) + 65536)
        (run_dir / "report.json").write_bytes(canonical(report))
        return report
