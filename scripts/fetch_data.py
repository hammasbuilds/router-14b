"""Pull 55,000 real human preference votes from the LMSYS Chatbot Arena.

    python scripts/fetch_data.py

**Why this dataset and not the other one.** The obvious corpus is
`lmsys/chatbot_arena_conversations` (33k votes) and it is **gated** — it returns
401 without a HuggingFace token, and this machine deliberately has none. An
earlier pass concluded from that single 401 that the whole category was closed
and went looking at 3k-row substitutes instead. That was wrong.
`lmarena-ai/arena-human-preference-55k` is the same kind of data, ungated, and
**larger**: 55,000 votes rather than 33,000.

**Why not RouterBench.** `withmartian/routerbench` is the obvious routing
benchmark and is also ungated, but it ships only `.pkl` files. Unpickling
executes arbitrary code at load time, so a downloaded pickle is a downloaded
program. It is not used here, and if it ever is, it should be inspected with
`pickletools.dis` first rather than handed to `read_pickle`.

101 MB, fetched in byte ranges and checked against Content-Length, because a
single GET of a file this size over this link truncates silently often enough
to matter.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
URL = (
    "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
OUT = DATA / "arena.parquet"
CHUNK = 4_000_000

# The published row count. A different number is a different corpus, and every
# figure computed from it would be about something else.
EXPECT_ROWS = 57_477


def expected_size(url: str) -> int:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "router-14b"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def fetch() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    total = expected_size(URL)

    if OUT.exists() and OUT.stat().st_size == total:
        print(f"  {OUT.name} already complete ({total / 1e6:.0f} MB)")
        return

    # Written to a .part and renamed only once the length checks out, so a
    # half-written file is never visible under its real name.
    #
    # The .part is RESUMED rather than restarted. The first version opened it
    # "wb", which truncates: this download died 1.5 MB short of 101 MB and a
    # retry would have re-fetched the whole thing. On a link that throttles to
    # near-nothing without warning, starting over is how a download never
    # finishes at all.
    partial = OUT.with_suffix(".parquet.part")
    written = partial.stat().st_size if partial.exists() else 0
    if written >= total:
        written = 0                       # a .part longer than the file is junk
    if written:
        print(f"  resuming at {written / 1e6:.0f} MB of {total / 1e6:.0f} MB ",
              end="", flush=True)
    else:
        print(f"  {OUT.name}  {total / 1e6:.0f} MB ", end="", flush=True)

    with partial.open("r+b" if written else "wb") as handle:
        handle.seek(written)
        while written < total:
            end = min(written + CHUNK, total) - 1
            request = urllib.request.Request(
                URL,
                headers={"User-Agent": "router-14b", "Range": f"bytes={written}-{end}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    block = response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"\n    failed at byte {written:,}: {exc}")
                raise
            if not block:
                raise OSError(f"{OUT.name}: empty response at byte {written:,}")
            handle.write(block)
            written += len(block)
            print(".", end="", flush=True)

    got = partial.stat().st_size
    if got != total:
        partial.unlink()
        raise OSError(f"{OUT.name}: got {got:,} bytes, expected {total:,}. Removed.")
    partial.replace(OUT)
    print(" ok")


def main() -> None:
    print("fetching Chatbot Arena human preferences (55k)")
    fetch()

    import pyarrow.parquet as pq

    table = pq.read_table(OUT)
    print(f"\nrows    {table.num_rows:,}")
    print(f"columns {table.column_names}")

    if table.num_rows != EXPECT_ROWS:
        print(
            f"\nEXPECTED {EXPECT_ROWS:,} rows. The upstream dataset has changed, and "
            "every number computed from it would be about a different corpus.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("\ncorpus ready")


if __name__ == "__main__":
    main()
