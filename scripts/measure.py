"""Every number in the README.

    python scripts/measure.py

No model is run. Every figure is counted off 57,477 human judgements, with the
model strength fitted on the training split and reported on all three.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router import corpus  # noqa: E402
from router import predictors as P  # noqa: E402


def rule(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def the_corpus() -> None:
    rule("the corpus")
    c = corpus.counts()
    print(f"human votes      {c['votes']:>8,}")
    print(f"models           {c['models']:>8}")
    print(f"decided          {c['decided']:>8,}   {c['decided'] / c['votes']:.1%}")
    print(f"ties             {c['ties']:>8,}   {c['ties'] / c['votes']:.1%}")
    train, val, test = corpus.splits()
    print(f"\nsplit 90/7/3     train {len(train):,}   val {len(val):,}   test {len(test):,}")


def the_tie_rate() -> None:
    rule("finding 1 — a third of the time nobody can tell")
    c = corpus.counts()
    print(f"  ties: {c['ties']:,} of {c['votes']:,}  ({c['ties'] / c['votes']:.1%})")
    print("\n  ^ that is the share of traffic where routing is free: two different")
    print("    models answered, a person read both, and could not choose. Any")
    print("    router that gets those to the cheaper model loses nothing that")
    print("    anybody noticed.")


def the_routing_number() -> None:
    rule("finding 2 — the weaker model is enough more than half the time")
    train, _, test = corpus.splits()
    strength = P.Strength.fit(train)
    rate, n = P.weaker_is_enough(test, strength)
    print(f"  test pairs where one model is weaker: {n:,}")
    print(f"  weaker model wins or draws:           {rate:.1%}")
    print(f"\n  {'how far apart the two models are':<32}{'n':>6}{'weaker is enough':>20}")
    for band, (r, count) in P.by_gap(test, strength).items():
        print(f"  {band:<32}{count:>6}{r:>19.1%}")
    print("\n  ^ the gradient is the check: the further apart two models are by")
    print("    training win rate, the less often the weaker one suffices. If that")
    print("    ran flat the strength estimate would be measuring nothing.")


def the_length_finding() -> None:
    rule("finding 3 — the longer answer wins, and length is not quality")
    train, val, test = corpus.splits()
    strength = P.Strength.fit(train)

    print(f"{'split':<8}{'longer wins':>14}{'n':>9}"
          f"{'stronger model wins':>22}{'n':>9}{'they agree':>13}")
    for name, split in (("train", train), ("val", val), ("test", test)):
        la, na = P.accuracy(split, P.longer)
        sa, ns = P.accuracy(split, strength.predict)
        ag, _ = P.agreement(split, P.longer, strength.predict)
        print(f"{name:<8}{la:>14.3f}{na:>9,}{sa:>22.3f}{ns:>9,}{ag:>13.3f}")

    la, _ = P.accuracy(test, P.longer)
    sa, _ = P.accuracy(test, strength.predict)
    print(f"\n  ^ counting characters predicts a human's preference {la:.1%} of the")
    print("    time. A coin would get 50%. The predictor has not read a word of")
    print("    either answer and does not know which models are competing.")
    print("\n    Knowing exactly which two models are in the fight — the whole")
    print(f"    leaderboard, fitted on {len(train):,} votes — is worth {sa:.3f}.")
    print("    The difference between knowing everything about the models and")
    print(f"    knowing nothing but the character count is {sa - la:.3f}.")


def what_this_costs() -> None:
    rule("what this means for a router, and what it cannot say")
    train, _, test = corpus.splits()
    strength = P.Strength.fit(train)
    rate, _ = P.weaker_is_enough(test, strength)
    bands = P.by_gap(test, strength)
    close = next(v for k, v in bands.items() if k.startswith("close"))

    print(f"  Route to the weaker model and {rate:.0%} of the time nobody minds.")
    print(f"  Between models that are close, {close[0]:.0%}.")
    print("\n  But a router trained to predict Arena preference is trained on a")
    print("  signal that a character count reproduces most of. Optimising it")
    print("  optimises partly for verbosity, and a router that learns to prefer")
    print("  long answers is not routing on quality — it is routing on length,")
    print("  and it will send work to whichever model rambles.")
    print("\n  THIS CORPUS HAS NO COSTS IN IT. No price, no latency, no parameter")
    print("  count. So the saving from routing cannot be computed here and is not")
    print("  claimed. What is measured is when the weaker model is good enough;")
    print("  what that is worth needs a price list this data does not contain.")


def main() -> None:
    the_corpus()
    the_tie_rate()
    the_routing_number()
    the_length_finding()
    what_this_costs()
    print()


if __name__ == "__main__":
    main()
