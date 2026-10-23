# MP2 Reflection

Ingest: 5 stories → 44 section-bounded chunks (target 500 chars, 80 overlap, hard cap 1000),
embedded with `text-embedding-3-small` into Qdrant; answers from `gpt-4o-mini`, k = 3.

| Run     | Retrieval setup                      | Source-match | Facts matched | Avg latency |
|:--------|:-------------------------------------|:------------:|:-------------:|:-----------:|
| Round 1 | Plain top-k vector search            |     5/5      |     14/22     |   ~3.2 s    |
| Round 2 | Metadata enrichment + sub-queries    |     5/5      |     17/22     |   ~7.5 s    |

| Question                | Round 1 facts | Round 2 facts | Still missing after round 2       |
|:------------------------|:-------------:|:-------------:|:----------------------------------|
| q1 Red-Headed League    |      3/5      |      3/5      | thief, white patches on trousers  |
| q2 Speckled Band        |      3/6      |      4/6      | ventilator, trained the snake     |
| q3 Blue Carbuncle       |      2/2      |      2/2      | —                                 |
| q4 Engineer's Thumb     |      4/4      |      4/4      | —                                 |
| q5 Scandal in Bohemia   |      2/5      |      4/5      | Godfrey Norton                    |

## Round 3: hybrid retrieval with a reranker

For round 3 I gave `retrieve()` a `mode` argument and ran `validate` once per mode on the same
collection. Simple is the round 1 setup and refined is the round 2 setup. Hybrid takes the top 20
from the vector search and the top 20 from BM25, fuses them with RRF, and keeps the top 3 after a
`ms-marco-MiniLM-L-6-v2` cross-encoder rerank. BM25 runs over the same metadata header + chunk
text that gets embedded.

| Mode    | Retrieval setup                      | Source-match | Facts matched | Avg latency |
|:--------|:-------------------------------------|:------------:|:-------------:|:-----------:|
| simple  | Plain top-k vector search            |     5/5      |     16/22     |   ~3.5 s    |
| refined | Metadata enrichment + sub-queries    |     5/5      |     16/22     |   ~7.1 s    |
| hybrid  | Vector + BM25, RRF, cross-encoder    |     5/5      |     14/22     |   ~4.7 s    |

| Question                | simple | refined | hybrid | What hybrid missed                       |
|:------------------------|:------:|:-------:|:------:|:-----------------------------------------|
| q1 Red-Headed League    |  3/5   |   2/5   |  1/5   | John Clay, thief, wanted for…, patches   |
| q2 Speckled Band        |  5/6   |   4/6   |  4/6   | swamp adder, trained the snake           |
| q3 Blue Carbuncle       |  2/2   |   2/2   |  2/2   | —                                        |
| q4 Engineer's Thumb     |  4/4   |   4/4   |  4/4   | —                                        |
| q5 Scandal in Bohemia   |  2/5   |   4/5   |  3/5   | Godfrey Norton, Charing Cross            |

Hybrid came last on facts. It was worst on q1, where it never put "The Identity of the
Assistant" in the top 3. The answer said Spaulding's real identity "is not explicitly stated".
I printed the hybrid candidates to see where it went wrong. RRF had done its job: that chunk was
tied for first after fusion. Then the cross-encoder scored it −5.5 and dropped it to fifth.
"The Strange Work", "Holmes Investigates" and "Introduction" all scored above +3. Those chunks
repeat the question's words (*assistant*, *pawnshop*, *Jabez Wilson*). The identity chunk
answers with *John Clay* and barely uses them. q2 had the same problem: "The Explanation",
which names the swamp adder, fell to fourth behind the investigation chunks. So the retrieval
part of hybrid worked and the reranker undid it. The reranker only sees the chunk text, not the
`title — section` label, and it was trained on short web passages, so it rewards word overlap
with the question.

The latency table hides a cold start. The first hybrid question has to build the BM25 index
and load the cross-encoder. That took 33 s in one run and 279 s in another, when it also
downloaded the model. The ~4.7 s average leaves that question out. After warm-up, hybrid is
about 1 s slower than simple but still faster than refined, because it makes no extra LLM call.

The scores also moved between runs with no code changes. Simple went from 14/22 in round 1 to
16/22 here, and refined went from 17/22 to 16/22. With only 22 facts, a 2-fact gap between
modes is within that noise.

## What worked
Splitting on the story's section headings before any size-based splitting. No chunk
crosses a section boundary, so every chunk carries a clean `title — section` label. That
label made citations specific ("The Red-Headed League — The Identity of the Assistant")
and kept source-match at 5/5 in both rounds: retrieval never picked up the wrong story.

Breaking multi-part questions into sub-queries in round 2 lifted fact recall from 14/22
to 17/22. The biggest gain was q5 (Irene Adler), which went from 2/5 to 4/5. It has three
parts (where, how, why gone), and round 1's single search only returned the "Irene Adler
Wins" chunk. Round 2 also retrieved "The Execution" chunk, which brought in the
smoke rocket and Charing Cross.

## What didn't work
Facts that sit in a different section from the main answer were still missed. For q5, Godfrey
Norton is named only in an earlier section ("Holmes Disguises Himself"). The answer says "her husband"
in both rounds because that chunk was never in the top 3. In the same way, q2 in round 2
sent its "how was it discovered" sub-query to "The Vigil" instead of "The Explanation".
So it lost *ventilator* and *trained the snake*, both of which round 1 had partly covered.
Decomposition helps when each part maps to one section. It can hurt when the sub-query
wording matches the narrative better than the explanation.

The round-2 answers also turned into bulleted Q&A lists, sometimes repeating the same fact
(q3 says "Peterson was a commissionaire" twice, q4 repeats Paddington). The answer prompt
mirrors the sub-queries too literally.

## What I'd change
Use a smaller, cheaper model for metadata extraction and query refinement, and
cache the refinement. Round 2 more than doubled latency (~3.2 s → ~7.5 s, worst case 8.9 s)
for +3 facts. I'd also raise k per sub-query so supporting details from other sections, like
Norton or the ventilator, can reach the context. The cross-encoder reranker I expected to fix
this did not help as-is (see the hybrid results above). It needs the section header in its input
or a stronger model before it beats plain RRF. Finally, I'd have the answer step write one
plain paragraph rather than echoing the sub-question structure.

## One surprise
How much the fact score depends on exact wording rather than correctness. In q1, round 2
retrieved the right chunk and quoted "wanted for murder, theft, smashing and forgery"
word for word. But it was marked missing *thief* because it wrote "one of the most cunning
thieves". Round 1 had written "thief" and was marked missing the "wanted for" phrase
instead. Both answers were correct and both scored 3/5. Neither ever mentioned the
"white patches on his trouser knees", even though that sentence is in the same retrieved chunk.
So the LLM summarises retrieved context selectively. Good retrieval is necessary but not
sufficient, and a substring-match grader under- and mis-reports quality in ways a
semantic or LLM-judge check would not.

I also expected hybrid (round 3) to do best, and it came last. BM25 plus a cross-encoder
sounded like a fix for the gaps in rounds 1 and 2: exact names like *John Clay* would match
literally, and the reranker would push the chunk that holds the answer to the top. Instead it
scored 14/22 against 16/22 for the other two modes. It lost most on q1, the question I thought
it would win. The reranker picked the chunks that sounded most like the question over the chunk
that answered it. Adding a stronger stage on top doesn't help if that stage scores on the wrong
signal.
