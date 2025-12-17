#!/usr/bin/env python3
"""
Update generation_config.json in each checkpoint so that `eos_token_id`
includes both the original EOS id(s) and the id for a specified EOS token string
(e.g., "<|im_end|>").

Behavior:
- For each checkpoint under CHECKPOINTS_ROOT (e.g., checkpoint-1234):
  * Read tokenizer_config.json (and fallback to tokenizer.json) to find the id of EOS_TOKEN.
  * Open generation_config.json and:
      - if "eos_token_id" is an int -> make it a list [old_int, new_id] (deduped).
      - if "eos_token_id" is a list -> append new_id if not already present.
  * Save changes in-place (a .bak backup is created).

You can set variables below or pass CLI args:
    python update_eos_ids.py --root /path/to/checkpoints --eos "<|im_end|>"
"""

import argparse
import json
import os
import sys
from glob import glob

# ======= Set these defaults (can override via CLI) =======
CHECKPOINTS_ROOT = "/path/to/your/merged-checkpoints"
EOS_TOKEN = "<|im_end|>"
# =========================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    # Write atomically with a backup
    bak = path + ".bak"
    if os.path.exists(path):
        try:
            os.replace(path, bak)
        except Exception:
            # Fallback: copy via read/write
            with open(path, "rb") as src, open(bak, "wb") as dst:
                dst.write(src.read())

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    
    try:
        os.remove(bak)
    except FileNotFoundError:
        pass

def find_token_id_in_tokenizer_config(tok_cfg_path, token_str):
    """
    Try to find token id in tokenizer_config.json.
    Handles a few common layouts:
      - "added_tokens_decoder": { "50256": {"content": "<|im_end|>", ...}, ... }
      - "eos_token": {"id": 2, "content": "</s>"} or "eos_token": "</s>" (not our target, but useful pattern)
      - "additional_special_tokens": ["<|im_end|>", ...] (ids may not be here)
    Returns int or None.
    """
    try:
        cfg = load_json(tok_cfg_path)
    except Exception:
        return None

    # 1) Scan added_tokens_decoder (id->object with "content")
    dec = cfg.get("added_tokens_decoder")
    if isinstance(dec, dict):
        for k, v in dec.items():
            # Some dumps store just the string, some store dicts
            # Normalize to a dict with 'content'
            content = None
            if isinstance(v, dict):
                content = v.get("content") or v.get("content_override") or v.get("content_original")
            elif isinstance(v, str):
                content = v
            if content == token_str:
                try:
                    return int(k)
                except Exception:
                    # Occasionally key is not numeric, ignore
                    pass

    # 2) Sometimes there's a direct map token -> id (rare). Attempt generic scan:
    #    Look for any dicts containing {"content": token_str, "id": N}
    def deep_find_id(obj):
        if isinstance(obj, dict):
            if obj.get("content") == token_str and "id" in obj and isinstance(obj["id"], int):
                return obj["id"]
            for vv in obj.values():
                r = deep_find_id(vv)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for item in obj:
                r = deep_find_id(item)
                if r is not None:
                    return r
        return None

    iid = deep_find_id(cfg)
    if isinstance(iid, int):
        return iid

    return None

def find_token_id_in_tokenizer_json(tok_json_path, token_str):
    """
    Fallback: scan tokenizer.json (common).
    Looks at "added_tokens": [{"id": N, "content": "<|im_end|>", ...}, ...]
    Also tries BPE vocab if present (rare for special chat tokens).
    """
    try:
        data = load_json(tok_json_path)
    except Exception:
        return None

    # 1) added_tokens
    added = data.get("added_tokens")
    if isinstance(added, list):
        for item in added:
            if isinstance(item, dict) and item.get("content") == token_str and isinstance(item.get("id"), int):
                return item["id"]

    # 2) Try model vocab (BPE): data["model"]["vocab"] : {token: id}
    model = data.get("model")
    if isinstance(model, dict):
        vocab = model.get("vocab")
        if isinstance(vocab, dict):
            vid = vocab.get(token_str)
            if isinstance(vid, int):
                return vid

    return None

def find_token_id_via_transformers(checkpoint_dir, token_str):
    """
    Last resort: use transformers to convert token->id.
    Only used if transformers is installed and previous methods fail.
    """
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        tid = tok.convert_tokens_to_ids(token_str)
        if isinstance(tid, int) and tid >= 0:
            return tid
    except Exception:
        pass
    return None

def get_token_id(checkpoint_dir, token_str):
    """
    Best-effort retrieval of token id for token_str from files in checkpoint_dir.
    Priority:
      tokenizer_config.json -> tokenizer.json -> transformers (if available)
    """
    tok_cfg = os.path.join(checkpoint_dir, "tokenizer_config.json")
    tok_json = os.path.join(checkpoint_dir, "tokenizer.json")

    # 1) tokenizer_config.json
    if os.path.isfile(tok_cfg):
        tid = find_token_id_in_tokenizer_config(tok_cfg, token_str)
        if isinstance(tid, int):
            return tid

    # 2) tokenizer.json
    if os.path.isfile(tok_json):
        tid = find_token_id_in_tokenizer_json(tok_json, token_str)
        if isinstance(tid, int):
            return tid

    # 3) transformers fallback
    tid = find_token_id_via_transformers(checkpoint_dir, token_str)
    if isinstance(tid, int):
        return tid

    return None

def update_generation_config(gen_cfg_path, new_token_id):
    """
    Update eos_token_id in generation_config.json:
      - int  -> [old_int, new_token_id] (deduped)
      - list -> append new_token_id if missing
      - missing -> set to [new_token_id]
    Returns (changed: bool, before, after)
    """
    data = load_json(gen_cfg_path)
    before = data.get("eos_token_id")

    if isinstance(before, int):
        after = [before]
        if new_token_id not in after:
            after.append(new_token_id)
        data["eos_token_id"] = after
        changed = True

    elif isinstance(before, list):
        after = list(before)
        if new_token_id not in after:
            after.append(new_token_id)
            data["eos_token_id"] = after
            changed = True
        else:
            changed = False

    else:
        # Field missing or another type: create a list with the new id
        data["eos_token_id"] = [new_token_id]
        changed = True

    if changed:
        save_json(gen_cfg_path, data)

    return changed, before, data.get("eos_token_id")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=CHECKPOINTS_ROOT, help="Root directory containing checkpoint-* subfolders")
    parser.add_argument("--eos", default=EOS_TOKEN, help="EOS token string to include (e.g., '<|im_end|>')")
    args = parser.parse_args()

    root = args.root
    eos_token = args.eos

    if not os.path.isdir(root):
        print(f"[ERROR] Root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Find checkpoint subdirs (checkpoint-*, global_step_*, or any dir)
    patterns = ["checkpoint-*", "global_step_*", "*"]
    seen = set()
    checkpoints = []
    for pat in patterns:
        for d in sorted(glob(os.path.join(root, pat))):
            if os.path.isdir(d) and d not in seen:
                seen.add(d)
                checkpoints.append(d)

    if not checkpoints:
        print(f"[INFO] No checkpoint directories found under {root}")
        return

    print(f"[INFO] Found {len(checkpoints)} checkpoint(s) under {root}")
    failures = 0
    updated = 0
    skipped = 0

    for ckpt in checkpoints:
        gen_cfg = os.path.join(ckpt, "generation_config.json")
        if not os.path.isfile(gen_cfg):
            print(f"  - {os.path.basename(ckpt)}: generation_config.json not found -> SKIP")
            skipped += 1
            continue

        token_id = get_token_id(ckpt, eos_token)
        if token_id is None:
            print(f"  - {os.path.basename(ckpt)}: could not resolve id for token '{eos_token}' -> SKIP")
            skipped += 1
            continue

        try:
            changed, before, after = update_generation_config(gen_cfg, token_id)
            if changed:
                print(f"  - {os.path.basename(ckpt)}: updated eos_token_id {before} -> {after}")
                updated += 1
            else:
                print(f"  - {os.path.basename(ckpt)}: eos_token_id already contains {token_id} -> NO CHANGE")
        except Exception as e:
            print(f"  - {os.path.basename(ckpt)}: update failed: {e}", file=sys.stderr)
            failures += 1

    print(f"\n[SUMMARY] updated={updated}, skipped={skipped}, failed={failures}")

if __name__ == "__main__":
    main()
