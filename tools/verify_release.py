"""Fast structural verification for the GitHub reproduction package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


EXPECTED = {
    "sim_pretrain_e294.ckpt": "49d7180ee114c7a68e4f521d34641bdb3c41a69545303ab5314a8ac508e951a3",
    "train120_rotstrat_e169.ckpt": "33450edfd1c78a4a595364517b74dc9e02ed13a7e392f1709a15c015af1bf82c",
    "train120_rotgeom_e19.ckpt": "6b538b067f90e37bfc7d8eb3bed41c2828d000e5a7263497ac69af9c045f9677",
    "train150_e4.ckpt": "62860acb95dde21087485745cd0d1cf43c98c25a283aacd2eee6bb5ba14dfbc1",
    "model_state_dict.pt": "01b118590702d41dfa045fc563ef0afd7dcc90cd4541662cad7843d945ff1993",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    weights = root / "weights"
    for name, expected in EXPECTED.items():
        path = weights / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {name}: {actual}")
        print(f"OK sha256 {name}")

    payload = torch.load(weights / "model_state_dict.pt", map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload)
    if len(state) != 226:
        raise RuntimeError(f"Expected 226 inference tensors, found {len(state)}")
    config = json.loads((weights / "model_state_dict.json").read_text())
    if config.get("output_type") != "coords":
        raise RuntimeError(f"Unexpected deployment config: {config}")
    print(f"OK inference checkpoint tensors={len(state)} config={config}")


if __name__ == "__main__":
    main()
