"""Against the real 57,477-vote corpus.

The test that matters most is `test_strength_is_fitted_on_train_only`: the
strength table is the only fitted thing in the project, and a leak from the
test split would make every number in the README a measurement of itself.
"""

from __future__ import annotations

import pytest

from router import corpus
from router import predictors as P

TRAIN, VAL, TEST = corpus.splits()
STRENGTH = P.Strength.fit(TRAIN)


def test_corpus_size():
    c = corpus.counts()
    assert c["votes"] == corpus.EXPECT_ROWS
    assert c["models"] == 64
    assert c["decided"] + c["ties"] == c["votes"]


def test_the_tie_rate_is_about_a_third():
    """The headline. If this moves, finding 1 is about a different corpus."""
    c = corpus.counts()
    assert 0.28 < c["ties"] / c["votes"] < 0.34


def test_every_vote_has_exactly_one_verdict():
    for v in corpus.load()[:5000]:
        assert v.winner in (corpus.A, corpus.B, corpus.TIE)


def test_splits_are_disjoint_and_complete():
    total = len(TRAIN) + len(VAL) + len(TEST)
    assert total == len(corpus.load())
    ids = lambda s: {id(v) for v in s}  # noqa: E731
    assert not ids(TRAIN) & ids(TEST)
    assert not ids(TRAIN) & ids(VAL)
    assert not ids(VAL) & ids(TEST)


def test_splits_are_stable():
    again = corpus.splits()
    assert [v.model_a for v in again[2][:20]] == [v.model_a for v in TEST[:20]]


def test_strength_is_fitted_on_train_only():
    """A strength table that had seen the test votes would be scoring itself.

    Fitting on the whole corpus and fitting on train must give different
    numbers; if they do not, the fit is not reading what it was given.
    """
    everything = P.Strength.fit(corpus.load())
    shared = set(STRENGTH.rate) & set(everything.rate)
    assert shared
    assert any(STRENGTH.rate[m] != everything.rate[m] for m in shared)


def test_a_model_with_too_few_games_has_no_rate():
    """A model with four games can sit at 1.000. That is not a win rate."""
    sparse = P.Strength.fit(TRAIN, min_games=10_000)
    assert len(sparse.rate) < len(STRENGTH.rate)
    for m in sparse.rate:
        assert sparse.games[m] >= 10_000


def test_ties_are_excluded_from_accuracy():
    """A predictor that must choose cannot be scored on a vote where the human
    refused to."""
    ties = [v for v in TEST if not v.decided]
    assert ties
    rate, seen = P.accuracy(ties, P.longer)
    assert seen == 0 and rate == 0.0


def test_predictors_abstain_rather_than_guess():
    vote = corpus.Vote("x", "y", corpus.A, 10, 100, 100)
    assert P.longer(vote) is None                    # equal lengths
    assert STRENGTH.predict(vote) is None            # unknown models


def test_longer_answer_beats_a_coin_flip():
    """Finding 3. Counting characters should not predict a human preference,
    and it does."""
    rate, n = P.accuracy(TEST, P.longer)
    assert n > 1000
    assert 0.55 < rate < 0.70


def test_model_identity_beats_length_but_barely():
    """The point of the finding: knowing the entire leaderboard is worth a few
    points over counting characters."""
    length, _ = P.accuracy(TEST, P.longer)
    strength, _ = P.accuracy(TEST, STRENGTH.predict)
    assert strength > length
    assert strength - length < 0.15


def test_the_two_predictors_are_not_the_same_signal():
    """If they agreed nearly always, length would just be a proxy for identity."""
    agree, n = P.agreement(TEST, P.longer, STRENGTH.predict)
    assert n > 1000
    assert agree < 0.75


def test_the_weaker_model_is_often_enough():
    """Finding 2, the routing number."""
    rate, n = P.weaker_is_enough(TEST, STRENGTH)
    assert n > 1000
    assert 0.45 < rate < 0.65


def test_the_weaker_model_helps_less_as_the_gap_grows():
    """The sanity check on the strength estimate.

    If this ran flat, the win rates would be sorting models into bands that do
    not correspond to anything a human noticed.
    """
    bands = P.by_gap(TEST, STRENGTH)
    rates = [r for r, _ in bands.values()]
    assert rates == sorted(rates, reverse=True), bands
    assert rates[0] - rates[-1] > 0.15


def test_length_gap_and_longer_agree():
    vote = corpus.Vote("x", "y", corpus.A, 10, 200, 100)
    assert vote.length_gap == 100
    assert vote.longer == corpus.A


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_every_split_has_both_ties_and_decisions(split):
    data = {"train": TRAIN, "val": VAL, "test": TEST}[split]
    assert any(v.decided for v in data)
    assert any(not v.decided for v in data)
