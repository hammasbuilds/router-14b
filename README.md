# router-14b

> **The weaker model is good enough 53.6% of the time.** And a predictor that counts characters — having read neither answer, and not knowing which models are competing — guesses the human's preference **61.7%** of the time.

**Status:** complete as a measurement. No model is run: every figure is counted off 57,477
human judgements, with the one fitted thing (model strength) fitted on the training split
and reported on all three.

## The corpus

[`lmarena-ai/arena-human-preference-55k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k)
— the Chatbot Arena shows a person one prompt and two anonymous answers and asks which is
better. They may also call it a tie.

| | |
|---|---:|
| Human votes | **57,477** |
| Models | 64 |
| Decided | 39,716 (69.1%) |
| **Ties** | **17,761 (30.9%)** |

Split 90/7/3 by vote, seeded: train 51,729 · val 4,023 · test 1,725.

```
python scripts/fetch_data.py   # 102 MB, ungated, no token
python scripts/measure.py      # every table below
python -m pytest               # 18 tests
```

## Finding 1 — a third of the time nobody can tell

**30.9% of votes are ties.** Two different models answered, a person read both, and could
not choose.

That is the share of traffic where routing is free. Any router that sends those to the
cheaper model loses nothing anybody noticed.

## Finding 2 — the weaker model is enough more than half the time

Model strength is each model's win rate, fitted on the 51,729 training votes. "Enough" means
the human either preferred the weaker model or could not tell them apart.

| | n | Weaker model is enough |
|---|---:|---:|
| **All test pairs** | **1,725** | **53.6%** |
| close (gap < 0.05) | 577 | 62.4% |
| middling (0.05–0.15) | 746 | 55.2% |
| far apart (> 0.15) | 402 | 38.1% |

The gradient is the check. The further apart two models are by training win rate, the less
often the weaker one suffices. If that ran flat, the strength estimate would be sorting
models into bands that correspond to nothing a human noticed.

## Finding 3 — the longer answer wins, and length is not quality

| Split | Longer wins | n | Stronger model wins | n | They agree |
|---|---:|---:|---:|---:|---:|
| train | 0.616 | 35,602 | 0.639 | 35,686 | 0.603 |
| val | 0.621 | 2,802 | 0.646 | 2,815 | 0.594 |
| **test** | **0.617** | 1,213 | **0.658** | 1,215 | 0.603 |

Counting characters predicts a human's preference **61.7%** of the time. A coin gets 50%.
That predictor has not read a word of either answer and does not know which models are
competing.

Knowing exactly which two models are in the fight — the whole leaderboard, fitted on 51,729
votes — is worth **0.658**.

**The difference between knowing everything about the models and knowing nothing but the
character count is 0.041.**

And the two are not the same signal: they pick the same side only 60.3% of the time, so the
5-point advantage is not length wearing a different hat.

## What this means for a router

Route to the weaker model and **54%** of the time nobody minds. Between models that are
close, **62%**.

But a router trained to predict Arena preference is trained on a signal that a character
count reproduces most of. Optimising it optimises partly for verbosity — and a router that
learns to prefer long answers is not routing on quality, it is routing on length, and it
will send work to whichever model rambles.

## What this cannot say

**There are no costs in this corpus.** No price, no latency, no parameter count. So the
saving from routing cannot be computed here and is not claimed. What is measured is *when
the weaker model is good enough*; what that is worth needs a price list this data does not
contain.

**Ties are excluded from accuracy.** A predictor that must choose a side cannot be scored on
a vote where the human refused to. Folding ties in as half credit flatters every predictor
equally and means nothing.

## The blocker that was not one

The obvious corpus is `lmsys/chatbot_arena_conversations` — 33,000 votes, and **gated**: it
returns 401 without a HuggingFace token, and this machine deliberately has none.

An earlier pass took that single 401 as proof the category was closed, looked at two 3k-row
substitutes, and shelved the project as blocked on data. That was an inference from one data
point. Probing eight candidates: **all eight are ungated**, and this one is the same kind of
data and **larger** — 57,477 votes against 33,000.

`withmartian/routerbench` is also ungated and is *not* used here: it ships only `.pkl` files,
and unpickling executes arbitrary code, so a downloaded pickle is a downloaded program. If it
is ever wanted it should go through `pickletools.dis` first, never `read_pickle`.

## Layout

```
scripts/fetch_data.py       102 MB, byte-ranged, resumable, length-checked
src/router/corpus.py        57,477 votes; text measured and discarded at load
src/router/predictors.py    strength (fitted on train), length, agreement
scripts/measure.py          every table above
tests/                      18 tests, incl. the train-only fit
```
