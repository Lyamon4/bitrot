"""bitrot MLX worker.

Speaks JSON lines on stdin/stdout. One request per line, one reply per line.
Started by the bitrot binary; not meant to be run by hand.
"""

import json
import os
import sys
import time


def send(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


try:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
except Exception as e:  # noqa: BLE001
    for _ in sys.stdin:
        send({"error": f"cannot import mlx / mlx_lm ({e}); run: pip install mlx-lm"})
    sys.exit(1)


def _mem_fn(name):
    # mlx >= 0.24 exposes these at top level, older versions under mx.metal.
    return getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)


get_peak_memory = _mem_fn("get_peak_memory")
reset_peak_memory = _mem_fn("reset_peak_memory")

model = None
tok = None


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith((".safetensors", ".npz")):
                total += os.path.getsize(os.path.join(root, f))
    return total


def op_load(req):
    global model, tok
    model, tok = load(req["path"])
    if reset_peak_memory:
        reset_peak_memory()
    path = req["path"]
    if not os.path.isdir(path):
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(path, local_files_only=True)
        except Exception:  # noqa: BLE001
            path = None
    n_params = 0
    try:
        from mlx.utils import tree_flatten

        for _, v in tree_flatten(model.parameters()):
            n_params += v.size
    except Exception:  # noqa: BLE001
        n_params = None
    logits = model(mx.array([[0]]))
    return {
        "n_vocab": int(logits.shape[-1]),
        "size_bytes": dir_size(path) if path else 0,
        "n_params": n_params,
        "device": f"MLX · {mx.default_device()}",
    }


def op_tokenize(req):
    return {"ids": tok.encode(req["text"], add_special_tokens=bool(req.get("add_special")))}


def op_pieces(req):
    ids = req["ids"]
    if not req.get("stream"):
        return {"pieces": [tok.decode([i]) for i in ids]}
    # Running text: attribute each character to the token that completes it,
    # so byte-split characters (Cyrillic, CJK, emoji) do not turn into U+FFFD.
    pieces, prev = [], ""
    for n in range(1, len(ids) + 1):
        text = tok.decode(ids[:n]).rstrip("�")
        if text.startswith(prev):
            pieces.append(text[len(prev):])
            prev = text
        else:
            pieces.append(tok.decode([ids[n - 1]]))
            prev = text
    return {"pieces": pieces}


def op_eval(req):
    ids = req["ids"]
    k = int(req["top_k"])
    logits = model(mx.array(ids)[None])[0, :-1].astype(mx.float32)
    lsm = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    top_idx = mx.argpartition(-lsm, kth=k - 1, axis=-1)[:, :k]
    top_lp = mx.take_along_axis(lsm, top_idx, axis=-1)
    order = mx.argsort(-top_lp, axis=-1)
    top_idx = mx.take_along_axis(top_idx, order, axis=-1)
    top_lp = mx.take_along_axis(top_lp, order, axis=-1)

    target = mx.take_along_axis(lsm, mx.array(ids[1:])[:, None], axis=-1)[:, 0]

    gathered = None
    if req.get("gather"):
        g = req["gather"]
        width = max(len(r) for r in g)
        padded = mx.array([r + [0] * (width - len(r)) for r in g])
        gathered = mx.take_along_axis(lsm, padded, axis=-1)
        mx.eval(gathered)
        gathered = [row[: len(r)] for row, r in zip(gathered.tolist(), g)]

    mx.eval(top_idx, top_lp, target)
    return {
        "top_ids": top_idx.tolist(),
        "top_lp": top_lp.tolist(),
        "target": target.tolist(),
        "gathered": gathered,
    }


def _run(ids, gen):
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(mx.array(ids)[None], cache=cache)
    y = mx.argmax(logits[0, -1])
    mx.eval(y)
    t1 = time.perf_counter()
    for _ in range(gen):
        logits = model(y.reshape(1, 1), cache=cache)
        y = mx.argmax(logits[0, -1])
        mx.eval(y)
    t2 = time.perf_counter()
    return len(ids) / (t1 - t0), gen / (t2 - t1)


def op_bench(req):
    _run(req["ids"][:16], 4)  # warm-up: compile kernels
    prefill, decode = _run(req["ids"], int(req["gen"]))
    return {"prefill_tok_s": prefill, "decode_tok_s": decode}


def op_chat_prompt(req):
    messages = [{"role": "user", "content": req["text"]}]
    if getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)
        if isinstance(ids, str):
            ids = tok.encode(ids, add_special_tokens=False)
    else:
        ids = tok.encode(req["text"])
    return {"ids": list(ids)}


def op_generate(req):
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    cache = make_prompt_cache(model)
    logits = model(mx.array(req["ids"])[None], cache=cache)
    out = []
    for _ in range(int(req["max_new"])):
        y = int(mx.argmax(logits[0, -1]).item())
        if y in eos:
            break
        out.append(y)
        logits = model(mx.array([[y]]), cache=cache)
    return {"ids": out}


def op_mem(_req):
    return {"peak_bytes": int(get_peak_memory()) if get_peak_memory else None}


OPS = {
    "load": op_load,
    "tokenize": op_tokenize,
    "pieces": op_pieces,
    "eval": op_eval,
    "bench": op_bench,
    "mem": op_mem,
    "chat_prompt": op_chat_prompt,
    "generate": op_generate,
}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("op") == "quit":
        break
    try:
        send(OPS[req["op"]](req))
    except Exception as e:  # noqa: BLE001
        send({"error": f"{type(e).__name__}: {e}"})
