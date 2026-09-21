"""57,477 head-to-head human votes between language models.

The Chatbot Arena shows a person one prompt and two anonymous answers and asks
which is better. They may also call it a tie. This is 57,477 of those judgements
across 64 models — the only large public record of humans comparing models
directly, rather than a benchmark comparing them to a reference answer.

    from router import corpus

    train, val, test = corpus.splits()
    train[0].winner        # "a" | "b" | "tie"
    train[0].length_gap    # characters, a minus b

**What is not in it.** There is no cost, no latency and no parameter count.
That bounds what this project can say: it can measure *when a weaker model is
good enough*, and it cannot measure what that saves. Anything about money would
have to come from a price list this corpus does not contain, and is not claimed
here.

**Why not the obvious dataset.** `lmsys/chatbot_arena_conversations` is the one
everybody cites and it is gated behind a HuggingFace token. An earlier pass took
that single 401 as proof the whole category was closed. This corpus is the same
kind of data, ungated, and larger — 57,477 votes against 33,000.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
ARENA = DATA / "arena.parquet"

A, B, TIE = "a", "b", "tie"

# 90/7/3. The corpus is 57k, which is the large end of the split policy: a 3%
# test set is still 1,725 votes, and holding back more of a corpus this size
# buys precision nobody needs at the cost of a worse strength estimate.
SHARES = (0.90, 0.07, 0.03)
SEED = 7

EXPECT_ROWS = 57_477


class CorpusMissingError(FileNotFoundError):
    """The Arena parquet is not on disk."""


@dataclass(frozen=True)
class Vote:
    """One human judgement between two models on one prompt."""

    model_a: str
    model_b: str
    winner: str
    prompt_len: int
    len_a: int
    len_b: int

    @property
    def decided(self) -> bool:
        """Whether the human picked one. A third of the time they did not."""
        return self.winner != TIE

    @property
    def length_gap(self) -> int:
        return self.len_a - self.len_b

    @property
    def longer(self) -> str | None:
        if self.len_a == self.len_b:
            return None
        return A if self.len_a > self.len_b else B


@lru_cache(maxsize=1)
def load() -> tuple[Vote, ...]:
    """Every vote. Text is measured and discarded.

    The parquet is 102 MB and almost all of it is response bodies. Nothing here
    needs to read them — only how long they are — so they are turned into two
    integers at load and never held.
    """
    if not ARENA.exists():
        raise CorpusMissingError(
            f"{ARENA} is missing. Run scripts/fetch_data.py, which pulls the "
            "Arena preferences from HuggingFace (ungated, no token)."
        )

    import pyarrow.parquet as pq

    table = pq.read_table(
        ARENA,
        columns=[
            "model_a", "model_b", "prompt", "response_a", "response_b",
            "winner_model_a", "winner_model_b", "winner_tie",
        ],
    )
    d = table.to_pydict()

    out = []
    for i in range(table.num_rows):
        if d["winner_tie"][i]:
            winner = TIE
        elif d["winner_model_a"][i]:
            winner = A
        elif d["winner_model_b"][i]:
            winner = B
        else:
            # The three flags are one-hot in this release; a row that is all
            # zero has no judgement in it and cannot be scored either way.
            continue
        out.append(
            Vote(
                model_a=d["model_a"][i],
                model_b=d["model_b"][i],
                winner=winner,
                prompt_len=len(d["prompt"][i] or ""),
                len_a=len(d["response_a"][i] or ""),
                len_b=len(d["response_b"][i] or ""),
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def splits() -> tuple[tuple[Vote, ...], tuple[Vote, ...], tuple[Vote, ...]]:
    """Train, validation, test — shuffled once from a fixed seed.

    Split by vote rather than by prompt. 84.6% of prompts appear exactly once,
    so there is almost nothing to leak, and the unit being predicted is the
    judgement rather than the prompt.
    """
    votes = list(load())
    random.Random(SEED).shuffle(votes)
    n = len(votes)
    a = int(n * SHARES[0])
    b = int(n * (SHARES[0] + SHARES[1]))
    return tuple(votes[:a]), tuple(votes[a:b]), tuple(votes[b:])


def models() -> set[str]:
    return {v.model_a for v in load()} | {v.model_b for v in load()}


def counts() -> dict[str, int]:
    votes = load()
    return {
        "votes": len(votes),
        "models": len(models()),
        "decided": sum(1 for v in votes if v.decided),
        "ties": sum(1 for v in votes if not v.decided),
    }
