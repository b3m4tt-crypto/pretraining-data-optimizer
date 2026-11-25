import argparse
import hashlib
from pathlib import Path

MIN_CHARS = 15
SPAM_PHRASES = ("click here",)


def canonical_hash(line: str) -> str:
    words = sorted(line.lower().split())
    canonical = " ".join(words)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def clean_file(
    input_path: Path,
    output_path: Path,
    log_path: Path,
    threshold: float = 0.7,
    min_chars: int = MIN_CHARS,
) -> dict[str, int]:
    seen_hashes: set[str] = set()
    kept_word_sets: list[set[str]] = []
    kept_lines: list[str] = []
    dropped_logs: list[str] = []

    counts = {
        "total_lines": 0,
        "kept_lines": 0,
        "empty": 0,
        "short": 0,
        "spam": 0,
        "exact_dupe": 0,
        "near_dupe": 0,
    }

    raw_lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    counts["total_lines"] = len(raw_lines)

    for raw in raw_lines:
        cleaned = " ".join(raw.strip().split())

        if not cleaned:
            counts["empty"] += 1
            dropped_logs.append(f"[EMPTY] {raw!r}")
            continue

        if len(cleaned) < min_chars:
            counts["short"] += 1
            dropped_logs.append(f"[SHORT] {cleaned}")
            continue

        lower_line = cleaned.lower()
        if any(phrase in lower_line for phrase in SPAM_PHRASES):
            counts["spam"] += 1
            dropped_logs.append(f"[SPAM] {cleaned}")
            continue

        h = canonical_hash(cleaned)
        if h in seen_hashes:
            counts["exact_dupe"] += 1
            dropped_logs.append(f"[EXACT_DUPE] {cleaned}")
            continue

        words = set(lower_line.split())
        is_near_dupe = any(
            jaccard_similarity(words, prev_words) >= threshold
            for prev_words in kept_word_sets
        )
        if is_near_dupe:
            counts["near_dupe"] += 1
            dropped_logs.append(f"[NEAR_DUPE] {cleaned}")
            continue

        seen_hashes.add(h)
        kept_word_sets.append(words)
        kept_lines.append(cleaned)
        counts["kept_lines"] += 1

    output_path.write_text(
        "\n".join(kept_lines) + ("\n" if kept_lines else ""),
        encoding="utf-8",
    )
    log_path.write_text(
        "\n".join(dropped_logs) + ("\n" if dropped_logs else ""),
        encoding="utf-8",
    )
    return counts


def main():
    parser = argparse.ArgumentParser(description="Clean and deduplicate text lines.")
    parser.add_argument("--input", required=True, help="Path to raw input text file")
    parser.add_argument("--output", required=True, help="Path to cleaned output text file")
    parser.add_argument("--log", default="dropped_lines.txt", help="Path to dropped-lines log")
    parser.add_argument("--threshold", type=float, default=0.7, help="Jaccard similarity cutoff")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS, help="Minimum line length")
    args = parser.parse_args()

    stats = clean_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        log_path=Path(args.log),
        threshold=args.threshold,
        min_chars=args.min_chars,
    )
    print("Cleaning summary:", stats)


if __name__ == "__main__":
    main()
