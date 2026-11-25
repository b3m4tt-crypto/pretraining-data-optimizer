import argparse
from pathlib import Path
import torch
import torch.nn as nn


class TinyNextTokenNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x):
        # x shape: (batch_size, block_size=16)
        emb = self.embed(x)
        return self.net(emb)


def run_bench(text: str, steps: int = 400, block_size: int = 16, batch_size: int = 32, lr: float = 1e-3):
    if len(text) <= block_size + 2:
        raise ValueError("Text is too short to train and validate.")

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    split = int(0.85 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    model = TinyNextTokenNet(vocab_size=len(chars))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def get_batch(source):
        max_start = len(source) - block_size - 1
        actual_batch = min(batch_size, max(1, max_start))
        ix = torch.randint(0, max_start, (actual_batch,))
        x = torch.stack([source[i : i + block_size] for i in ix])
        y = torch.stack([source[i + block_size] for i in ix])
        return x, y

    @torch.no_grad()
    def eval_loss():
        model.eval()
        if len(val_data) <= block_size + 1:
            return float("nan")
        x_val, y_val = get_batch(val_data)
        logits = model(x_val)
        loss = criterion(logits, y_val).item()
        model.train()
        return loss

    start_val = eval_loss()
    for _ in range(steps):
        xb, yb = get_batch(train_data)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_val = eval_loss()
    return {
        "total_chars": len(text),
        "train_chars": len(train_data),
        "val_chars": len(val_data),
        "start_val_loss": round(start_val, 4),
        "final_val_loss": round(final_val, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Equal-step micro-pretraining benchmark.")
    parser.add_argument("--raw", required=True, help="Path to raw text file")
    parser.add_argument("--filtered", required=True, help="Path to cleaned/filtered text file")
    parser.add_argument("--steps", type=int, default=400, help="Training steps per run")
    args = parser.parse_args()

    raw_text = Path(args.raw).read_text(encoding="utf-8", errors="ignore")
    filtered_text = Path(args.filtered).read_text(encoding="utf-8", errors="ignore")

    raw_stats = run_bench(raw_text, steps=args.steps)
    filt_stats = run_bench(filtered_text, steps=args.steps)

    print("RAW:", raw_stats)
    print("FILTERED:", filt_stats)


if __name__ == "__main__":
    main()
