"""AlignScore 议题一致性评测（ACL 2023）."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNScore_SRC = ROOT / "third_party" / "AlignScore" / "src"
DEFAULT_CKPT = ROOT / "checkpoints" / "AlignScore" / "AlignScore-base.ckpt"


class AlignScoreEvaluator:
    def __init__(
        self,
        *,
        ckpt_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 8,
    ):
        if str(ALIGNScore_SRC) not in sys.path:
            sys.path.insert(0, str(ALIGNScore_SRC))
        from alignscore import AlignScore

        ckpt = Path(ckpt_path or DEFAULT_CKPT)
        if not ckpt.exists():
            raise FileNotFoundError(f"AlignScore checkpoint not found: {ckpt}")

        dev = device
        if dev == "cpu":
            dev_arg = "cpu"
        elif dev.startswith("cuda"):
            dev_arg = dev
        else:
            dev_arg = f"cuda:{dev}" if dev.isdigit() else dev

        self.scorer = AlignScore(
            model="roberta-base",
            batch_size=batch_size,
            device=dev_arg,
            ckpt_path=str(ckpt),
            evaluation_mode="nli_sp",
            verbose=False,
        )

    def score_issue_consistency(self, issue: str, generated_text: str) -> float:
        scores = self.scorer.score([issue], [generated_text])
        return float(scores[0])
