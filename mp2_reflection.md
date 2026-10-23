# MP2 Reflection

Ingest: 5 stories → 44 section-bounded chunks (target 500 chars, 80 overlap, hard cap 1000),
embedded with `text-embedding-3-small` into Qdrant; answers from `gpt-4o-mini`, k = 3.
All numbers below come from `mp2_validation.txt`: one `validate` run per retrieval mode, over the
2 predefined and 3 learner questions (22 expected facts in total).

| Run     | Retrieval setup                                   | Source-match | Facts matched | Avg latency          |
|:--------|:--------------------------------------------------|:------------:|:-------------:|:--------------------:|
| Simple  | Plain top-k vector search                         |     5/5      |     16/22     | ~3.9 s               |
| Refined | Metadata enrichment + sub-queries + RRF           |     5/5      |     17/22     | ~8.0 s               |
| Hybrid  | Dense + BM25 → RRF → cross-encoder rerank         |     5/5*     |     14/22     | ~6.9 s (~4.1 s warm) |

\* Hybrid's q4 also cited `03_blue_carbuncle.txt`. The grader only checks that the expected source
is somewhere in the citations, so it still counts as a hit, but it is the only run where a chunk
from the wrong story got into the top 3.

| Question                | Simple | Refined | Hybrid | Missing in the best run                     |
|:------------------------|:------:|:-------:|:------:|:--------------------------------------------|
| q1 Red-Headed League    |  3/5   |   2/5   |  1/5   | thief, white patches on his trouser knees   |
| q2 Speckled Band        |  5/6   |   5/6   |  4/6   | swamp adder (simple) / trained the snake (refined) |
| q3 Blue Carbuncle       |  2/2   |   2/2   |  2/2   | —                                           |
| q4 Engineer's Thumb     |  4/4   |   4/4   |  4/4   | —                                           |
| q5 Scandal in Bohemia   |  2/5   |   4/5   |  3/5   | Godfrey Norton                              |

`validate` calls `answer(q, k=3)` without a mode, so each run was made by changing the default
`mode` on `answer()`. The hybrid run really used hybrid: the log shows the cross-encoder weights
loading before q1, and there are no reworded sub-questions in the answers, which only refined
mode produces.

Latencies per question (ms):

| Mode    | q1     | q2    | q3    | q4    | q5    |
|:--------|-------:|------:|------:|------:|------:|
| Simple  |  4348  | 5847  | 3450  | 2766  | 2924  |
| Refined |  9544  | 8592  | 7000  | 6432  | 8599  |
| Hybrid  | 18369  | 3657  | 4660  | 3174  | 4710  |

## Round 3: hybrid retrieval

Hybrid takes the top 20 from vector search and the top 20 from BM25, fuses them with RRF, and
keeps the top 3 after a `ms-marco-MiniLM-L-6-v2` cross-encoder rerank. It is selected with a new
`mode` argument on `retrieve()`.

**Result: hybrid scored lowest, 14/22.** It tied on q3 and q4 and lost on the three questions
that need a specific section:

- **q1 (1/5).** It missed "The Identity of the Assistant", the only chunk naming John Clay,
  which both other modes retrieved. The question's words (*assistant*, *Jabez Wilson*,
  *pawnshop*) fill the narrative chunks, so BM25 and the reranker favoured those. The answer
  honestly said the identity "is not explicitly stated", but only *Vincent Spaulding* matched.
- **q2 (4/6).** It never reached "The Explanation" or "The Vigil", so *swamp adder* and
  *trained the snake* were both missing.
- **q5 (3/5).** "The Smoke and Fire Plan" gave the best account of the trick of any run. But its
  "Irene Adler Wins" chunk didn't include the Charing Cross train that refined quoted.

**Latency.** The first question took 18.4 s (BM25 index build + cross-encoder load). After that
it averaged ~4.1 s, close to simple (~3.9 s) and half of refined (~8.0 s). No extra LLM calls.

**Answer format.** With no sub-queries, the model fills the system prompt's checklist literally
as **Who / What / Where / When / How / Evidence / Motive** bullets. This pads answers and invents
fields: q3's "When" is "Two mornings after Christmas, specifically on Christmas morning", and q1's
"Motive" is "not detailed". The checklist helps refined mode but hurts hybrid.

## W9 stretch goal: BM25 + RRF

Implemented in `_hybrid_rerank_retrieve()` in `src/mp2_rag.py`. BM25 indexes the same 44 chunks
as the vectors, using `embedding_text()` (metadata header + chunk body), so a query can match on
tags like "Godfrey Norton" as well as the prose. `filters` are mirrored in Python so both
retrievers see the same subset, and each result keeps its `rrf_score` next to the rerank `score`.

**Things I'd tidy up.**
- **Rank offset.** Refined counts RRF ranks from 0 and hybrid from 1. Tiny effect (1/60 vs 1/61),
  but they should agree.
- **Reranker misses the header.** The cross-encoder scores only the chunk body. In q1 it never saw
  the John Clay tag on "The Identity of the Assistant".
- **Coarse IDF.** With 44 chunks, a term like "holmes" appears in over half of them and its IDF is
  floored near zero, so it adds almost nothing.
- **Raw question.** Hybrid gets none of refined's story targeting or decomposition, which is why
  it does worst on multi-part questions.

## What worked
- **Section-bounded chunks.**
  - Every chunk has a clean `title — section` label.
  - Citations are specific ("The Red-Headed League — The Identity of the Assistant").
  - Source-match 5/5 in every mode.
- **Sub-queries (refined).**
  - Best total, 17/22. All of the gain is q5 (2/5 → 4/5).
  - "Where/how" added "The Execution" (*smoke rocket*).
  - "Why gone" got the Charing Cross line.
- **"Reuse the excerpts' exact wording."**
  - Simple's q1 quoted "wanted for murder, theft, smashing and forgery" verbatim.
  - Its q2 wrote "trained the snake".

## What didn't work
- **Facts in other sections.** Missed in every mode.
  - q5: never names Godfrey Norton, only "her husband".
  - q1: no mode got "white patches on his trouser knees".
- **Decomposition can hurt.**
  - q2 refined: went to "The Vigil", got *swamp adder*, but wrote "was trained by".
  - q2 simple: the reverse. Both 5/6.
  - q1 refined: 2/5 vs simple's 3/5. Dropped the "wanted for…" line.
- **Repetitive answers.**
  - Refined repeats facts: q3 commissionaire, q4 Paddington, each twice.
  - Hybrid filled the checklist (see above).
- **Open questions in `ask`.**
  - "Who is Sherlock Holmes?": misread him as "THE woman".
  - Bohemia vs Speckled Band: all 3 chunks from one story.
  - One top-3 search can't cover two stories.

## What I'd change
- **Give hybrid the refined front end.**
  - Run hybrid per sub-query, with the story filter.
  - Should keep refined's q5 gain at near-hybrid latency.
- **Rerank on the metadata header too.**
  - So tagged names (John Clay, Godfrey Norton) count.
- **Raise k per sub-query.**
  - Lets details from other sections (Norton, ventilator) in.
- **Cheaper, cached refinement model.**
  - Refined doubles latency (~3.9 s → ~8.0 s) for +1 fact.
- **One plain-paragraph answer.**
  - Drop the who/what/where checklist and sub-question echo.
- **Run each mode several times.**
  - Gaps here are only 1–3 facts out of 22.

## One surprise
- **The grader scores wording, not correctness.**
  - "thieves" doesn't contain "thief", so correct answers missed it.
  - "was trained by" missed *trained the snake*.
  - Hybrid's q1 was the most honest answer and scored lowest.
  - A semantic or LLM-judge check would catch these.
- **Hybrid came last.** I expected it to win.
  - The answers are split across sections, and BM25 + rerank doesn't fix that.
  - Targeting the story and splitting the question mattered more.
