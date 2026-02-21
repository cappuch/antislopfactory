import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import aiohttp
import numpy as np
from datasets import load_dataset

from util.classifier import MLPClassifier
from util.embed import embed

CATEGORY_MAP: dict[str | None, str] = { # dirty map lol
    "violence": "violence",
    "Violence": "violence",
    "Physical Harm": "violence",
    "hate": "hate",
    "Hate Speech": "hate",
    "Hate/Identity Hate": "hate",
    "harassment": "harassment",
    "Harassment": "harassment",
    "sexual": "sexual",
    "Pornography": "sexual",
    "self-harm": "self_harm",
    "Self-Harm": "self_harm",
    "self harm": "self_harm",
    "illegal": "illegal",
    "Illegal Activity": "illegal",
    "Criminal Planning/Confessions": "criminal",
    "Malware": "malware",
    "Fraud": "fraud",
    "privacy": "privacy",
    "Privacy Violence": "privacy",
    "PII/Privacy": "privacy",
    "deception": "deception",
    "harmful": "harmful",
    "unethical": "unethical",
    "Legal Opinion": "professional_advice",
    "Financial Advice": "professional_advice",
    "Health Consultation": "professional_advice",
    "Gov Decision": "professional_advice",
    "Political Lobbying": "professional_advice",
    "Economic Harm": "professional_advice",
    # borderline or usm
    "Needs Caution": "needs_caution",
}

KEYWORD_FALLBACKS: list[tuple[str, str]] = [
    ("violen", "violence"),
    ("hate", "hate"),
    ("harass", "harassment"),
    ("sex", "sexual"),
    ("porn", "sexual"),
    ("self-harm", "self_harm"),
    ("self harm", "self_harm"),
    ("suicid", "self_harm"),
    ("illegal", "illegal"),
    ("criminal", "criminal"),
    ("malware", "malware"),
    ("fraud", "fraud"),
    ("priva", "privacy"),
    ("decep", "deception"),
    ("harm", "harmful"),
    ("ethic", "unethical"),
]


def normalise_category(raw: str | None, label: int | None) -> str:
    if label == 0:
        return "safe"
    if raw is None:
        return "unsafe_other"
    if raw in CATEGORY_MAP:
        return CATEGORY_MAP[raw]
    low = raw.lower()
    for keyword, cat in KEYWORD_FALLBACKS:
        if keyword in low:
            return cat
    return "unsafe_other"

CACHE_DIR = Path("data/embeddings")
MAX_CONCURRENT = 64


async def _embed_one_batch(sem: asyncio.Semaphore, session: aiohttp.ClientSession,
                           batch: list[str], idx: int) -> tuple[int, np.ndarray | None]:
    """Embed a single batch with retries, respecting the semaphore."""
    async with sem:
        for attempt in range(3):
            try:
                vecs = await embed(batch, session=session)
                return idx, np.array(vecs, dtype=np.float32)
            except Exception as e:
                wait = 2 ** attempt
                print(f"\n  embed error at batch {idx} ({e}), retrying in {wait}s...")
                await asyncio.sleep(wait)
        print(f"\n  skipping batch {idx} after 3 failures")
        return idx, None


async def embed_batched_async(texts: list[str], batch_size: int = 32,
                              concurrency: int = MAX_CONCURRENT,
                              cache_path: Path | None = None,
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (embeddings, kept_indices) — indices into the original texts list."""
    if cache_path and cache_path.exists():
        print(f"  loading cached embeddings from {cache_path}")
        d = np.load(cache_path)
        return d["X"], d["idx"]

    texts = [t if t else " " for t in texts]
    total = len(texts)
    sem = asyncio.Semaphore(concurrency)
    done_count = 0
    t0 = time.time()

    def _progress(_fut):
        nonlocal done_count
        done_count += 1
        done_items = min(done_count * batch_size, total)
        elapsed = time.time() - t0
        rate = done_items / elapsed if elapsed > 0 else 0
        eta = (total - done_items) / rate if rate > 0 else 0
        print(f"\r  embedded {done_items:>7}/{total}  ({rate:.0f}/s  ETA {eta:.0f}s)", end="", flush=True)

    async with aiohttp.ClientSession() as session:
        tasks: list[asyncio.Task] = []
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            t = asyncio.create_task(_embed_one_batch(sem, session, batch, i))
            t.add_done_callback(_progress)
            tasks.append(t)

        results = await asyncio.gather(*tasks)

    print()
    results.sort(key=lambda r: r[0])
    skipped = sum(1 for _, v in results if v is None)
    if skipped:
        print(f"  skipped {skipped} batches ({skipped * batch_size} texts)")

    kept_vecs = []
    kept_idx = []
    for batch_start, vecs in results:
        if vecs is None:
            continue
        kept_vecs.append(vecs)
        kept_idx.extend(range(batch_start, batch_start + len(vecs)))

    X = np.concatenate(kept_vecs, axis=0)
    idx = np.array(kept_idx)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, idx=idx)
        print(f"  cached embeddings to {cache_path}")

    return X, idx


def embed_batched(texts: list[str], batch_size: int = 32,
                  concurrency: int = MAX_CONCURRENT,
                  cache_path: Path | None = None,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Sync entry point — runs the async embedder. Returns (X, kept_indices)."""
    return asyncio.run(embed_batched_async(texts, batch_size, concurrency, cache_path))

def main():
    parser = argparse.ArgumentParser(description="Train safety classifier")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="cap train samples (0 = all)")
    parser.add_argument("--test-count", type=int, default=0,
                        help="cap test samples (0 = all)")
    parser.add_argument("--skip-embed", action="store_true",
                        help="(deprecated, caching is automatic)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="max concurrent embed API requests")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="fraction of train data to hold out for validation")
    parser.add_argument("--out", type=str, default="data/safety_classifier.npz")
    args = parser.parse_args()

    print("loading dataset...")
    ds = load_dataset("SalKhan12/prompt-safety-dataset")
    train_ds = ds["train"]
    test_ds = ds["test"]

    if args.max_samples > 0:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
    if args.test_count > 0:
        test_ds = test_ds.select(range(min(args.test_count, len(test_ds))))

    print("normalising categories...")
    train_labels = [normalise_category(r["content_category"], r["label"]) for r in train_ds]
    test_labels = [normalise_category(r["content_category"], r["label"]) for r in test_ds]

    all_labels = sorted(set(train_labels))
    print(f"  {len(all_labels)} classes: {all_labels}")
    print("  train distribution:")
    for cat, n in Counter(train_labels).most_common():
        print(f"    {cat:<20s} {n:>7}")

    train_suffix = f"_{args.max_samples}" if args.max_samples else ""
    test_suffix = f"_{args.test_count}" if args.test_count else ""
    train_cache = CACHE_DIR / f"train{train_suffix}.npz"
    test_cache = CACHE_DIR / f"test{test_suffix}.npz"

    print("embedding train texts...")
    X_full, train_kept = embed_batched(train_ds["text"], args.embed_batch_size, args.concurrency, train_cache)
    train_labels = [train_labels[i] for i in train_kept]

    print("embedding test texts...")
    X_test, test_kept = embed_batched(test_ds["text"], args.embed_batch_size, args.concurrency, test_cache)
    test_labels = [test_labels[i] for i in test_kept]

    all_labels = sorted(set(train_labels))
    print(f"  {len(all_labels)} classes after filtering: {all_labels}")

    n_full = len(X_full)
    n_val = int(n_full * args.val_split)
    n_train = n_full - n_val

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_full)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    X_train = X_full[train_idx]
    X_val = X_full[val_idx]
    y_train = [train_labels[i] for i in train_idx]
    y_val = [train_labels[i] for i in val_idx]

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    # ── train ────────────────────────────────────────────────────
    print(f"\ntraining MLP (512 → {args.hidden_dim} → {len(all_labels)})...")
    clf = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        labels=all_labels,
    )
    history = clf.train(X_train, y_train,
                        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                        X_val=X_val, y_val=y_val)

    print("\nevaluating on val set...")
    val_preds = clf.predict(X_val)
    val_correct = sum(p == t for p, t in zip(val_preds, y_val))
    val_acc = val_correct / len(y_val)
    print(f"  val accuracy: {val_acc:.4f} ({val_correct}/{len(y_val)})")

    print("\nevaluating on test set...")
    preds = clf.predict(X_test)
    correct = sum(p == t for p, t in zip(preds, test_labels))
    acc = correct / len(test_labels)
    print(f"  accuracy: {acc:.4f} ({correct}/{len(test_labels)})")

    print("\n  per-class:")
    for cat in all_labels:
        cat_mask = [t == cat for t in test_labels]
        cat_total = sum(cat_mask)
        if cat_total == 0:
            continue
        cat_correct = sum(p == t for p, t, m in zip(preds, test_labels, cat_mask) if m)
        print(f"    {cat:<20s} {cat_correct:>5}/{cat_total:<5}  ({cat_correct/cat_total:.3f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(out))
    print(f"\nmodel saved to {out}")

    per_class = {}
    for cat in all_labels:
        cat_total = sum(1 for t in test_labels if t == cat)
        if cat_total == 0:
            continue
        cat_correct = sum(1 for p, t in zip(preds, test_labels) if t == cat and p == t)
        per_class[cat] = {"correct": cat_correct, "total": cat_total,
                          "accuracy": cat_correct / cat_total}

    log = {
        "args": vars(args),
        "labels": all_labels,
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(test_labels),
        "history": history,
        "val_accuracy": val_acc,
        "test_accuracy": acc,
        "test_per_class": per_class,
    }
    log_path = out.with_suffix(".json")
    log_path.write_text(json.dumps(log, indent=2))
    print(f"logs saved to {log_path}")


if __name__ == "__main__":
    main()
