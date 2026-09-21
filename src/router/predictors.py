"""Three ways to guess which answer a human preferred, and what each knows.

The routing question is *when will the cheaper model do*. Before that can be
answered, something has to predict what a person will prefer — and the point of
this module is that the two obvious predictors know very different things and
score almost the same.

* **`Strength`** — fits each model's win rate on the training votes and picks
  the model with the better record. It knows which two models are in the fight
  and nothing about what they said.
* **`longer`** — picks whichever answer has more characters. It knows nothing
  about either model and has not read a word.
* **`agreement`** — how often the two pick the same side, which turns out to be
  barely more often than chance would give.

Ties are excluded from accuracy throughout. A predictor that must choose a side
cannot be scored on a vote where the human refused to, and folding ties in as
half-credit flatters every predictor equally while meaning nothing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .corpus import A, B, Vote

# A model needs this many decided training votes before its win rate is used.
# Below it the rate is noise: a model with four games can sit at 1.000.
MIN_GAMES = 50


@dataclass
class Strength:
    """Each model's win rate, fitted on training votes only."""

    rate: dict[str, float]
    games: dict[str, int]

    @classmethod
    def fit(cls, votes, min_games: int = MIN_GAMES) -> Strength:
        wins: Counter = Counter()
        games: Counter = Counter()
        for v in votes:
            games[v.model_a] += 1
            games[v.model_b] += 1
            if v.decided:
                wins[v.model_a if v.winner == A else v.model_b] += 1
        rate = {m: wins[m] / games[m] for m in games if games[m] >= min_games}
        return cls(rate=rate, games=dict(games))

    def knows(self, vote: Vote) -> bool:
        return vote.model_a in self.rate and vote.model_b in self.rate

    def predict(self, vote: Vote) -> str | None:
        """The stronger model, or None when they are unknown or equal."""
        if not self.knows(vote):
            return None
        a, b = self.rate[vote.model_a], self.rate[vote.model_b]
        if a == b:
            return None
        return A if a > b else B

    def gap(self, vote: Vote) -> float | None:
        if not self.knows(vote):
            return None
        return abs(self.rate[vote.model_a] - self.rate[vote.model_b])

    def weaker(self, vote: Vote) -> str | None:
        strong = self.predict(vote)
        if strong is None:
            return None
        return B if strong == A else A


def longer(vote: Vote) -> str | None:
    """The longer answer, or None when they are the same length."""
    return vote.longer


def accuracy(votes, predict) -> tuple[float, int]:
    """Share correct over the votes this predictor will commit to.

    Returns the rate and the number of votes it was computed over, because the
    two predictors here abstain on different rows and comparing rates without
    the denominators would be comparing two different questions.
    """
    right = seen = 0
    for v in votes:
        if not v.decided:
            continue
        guess = predict(v)
        if guess is None:
            continue
        seen += 1
        right += guess == v.winner
    return (right / seen if seen else 0.0), seen


def agreement(votes, first, second) -> tuple[float, int]:
    """How often two predictors pick the same side."""
    same = seen = 0
    for v in votes:
        a, b = first(v), second(v)
        if a is None or b is None:
            continue
        seen += 1
        same += a == b
    return (same / seen if seen else 0.0), seen


def weaker_is_enough(votes, strength: Strength) -> tuple[float, int]:
    """How often the weaker of the two models wins or draws.

    This is the routing number. 'Enough' means the human either preferred the
    weaker model or could not tell them apart — both are outcomes where sending
    the request to the cheaper model cost nothing anybody noticed.
    """
    ok = seen = 0
    for v in votes:
        weak = strength.weaker(v)
        if weak is None:
            continue
        seen += 1
        ok += (not v.decided) or v.winner == weak
    return (ok / seen if seen else 0.0), seen


def by_gap(votes, strength: Strength, edges=(0.05, 0.15)) -> dict[str, tuple[float, int]]:
    """`weaker_is_enough`, split by how far apart the two models are.

    The routing decision is not the same question for two models that are
    neck-and-neck and for one that is far better. Reporting a single rate over
    both hides exactly the thing a router needs to know.
    """
    bands: dict[str, list[int]] = {
        f"close (<{edges[0]})": [0, 0],
        f"middling ({edges[0]}-{edges[1]})": [0, 0],
        f"far apart (>{edges[1]})": [0, 0],
    }
    names = list(bands)
    for v in votes:
        weak = strength.weaker(v)
        if weak is None:
            continue
        gap = strength.gap(v)
        band = names[0] if gap < edges[0] else names[1] if gap < edges[1] else names[2]
        bands[band][1] += 1
        bands[band][0] += (not v.decided) or v.winner == weak
    return {k: (ok / n if n else 0.0, n) for k, (ok, n) in bands.items()}
