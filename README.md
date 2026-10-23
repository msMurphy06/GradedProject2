**Repository:** https://github.com/msMurphy06/GradedProject2

# GradedProject2 — Mini-RAG over Sherlock Holmes

A retrieval-augmented generation (RAG) pipeline that answers questions about five Sherlock Holmes stories and cites the story and section each answer comes from.

```
corpus/*.txt  →  chunks  →  metadata + embeddings  →  Qdrant
                                                        ↓
                          question  →  retrieve  →  answer + citations
```

Three retrieval strategies can be compared on the same collection:

| Mode | Idea |
|---|---|
| `simple` | A single top-k cosine search |
| `refined` (default) | An LLM picks the story and splits the question into metadata-tagged sub-queries. Each sub-query's plain and metadata-filtered searches are fused with RRF |
| `hybrid` | Top 20 dense + top 20 BM25, fused with RRF, then reranked by a `ms-marco-MiniLM-L-6-v2` cross-encoder |

## Project structure

```
GradedProject2/
├── src/
│   └── mp2_rag.py                 # The whole pipeline: ingest, retrieve, answer, validate, CLI
├── corpus/                        # 5 stories (Red-Headed League, Speckled Band, Blue Carbuncle,
│                                  #   Engineer's Thumb, Scandal in Bohemia)
├── data/
│   ├── predefined_questions.jsonl # 2 provided questions (q1, q2) with expected source and facts
│   └── learner_questions.jsonl    # 3 questions I wrote (q3–q5)
├── mp2_reflection.md              # Results and analysis across the three rounds
├── requirements.txt
├── .env                           # Secrets and config (git-ignored)
```

## Setup

1. Create the virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows (source .venv/bin/activate on macOS/Linux)
   pip install -r requirements.txt
   ```

   `rank-bm25` and `sentence-transformers` are only needed for `hybrid` mode.

2. Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=<your key>
   OPENAI_BASE_URL=<optional, e.g. a proxy URL>
   QDRANT_URL=<your Qdrant URL>
   QDRANT_API_KEY=<your Qdrant key>
   ```

3. Load it into the environment. `mp2_rag.py` reads these from environment variables and does not load `.env` itself:

   ```bash
   set -a; source .env; set +a      # bash
   ```

   ```powershell
   Get-Content .env | ForEach-Object { if ($_ -match '^\s*([^#=]+)=(.*)$') { Set-Item "env:$($matches[1].Trim())" $matches[2].Trim() } }   # PowerShell
   ```

## Run

```bash
python src/mp2_rag.py ingest     # chunk, enrich, embed and load the corpus into Qdrant (run once)
python src/mp2_rag.py ask        # interactive Q&A loop; empty line or Ctrl-C to exit
python src/mp2_rag.py validate   # run both question files and score them
```

`ingest` recreates the `Sherlock_collection` collection each time, so re-running it starts from an empty index.

## How `mp2_rag.py` works

1. **Load** (`load_corpus`): reads each story. Its first non-empty line is the title.
2. **Chunk** (`chunk_document`): splits on the story's section headings first, so no chunk crosses a section boundary. Each section is then split recursively (paragraphs → lines → sentences → words) to a hard cap of 1000 characters. The 5 stories give 44 chunks, each labelled `title — section`.
3. **Enrich** (`enrich_metadata`): one `gpt-4o-mini` call per chunk, run 8 at a time, extracts `characters`, `places`, `key_items`, `occurrence_date` and `occurrence_time`. Provenance fields (`chunk_id`, `source`, `version`, `ingested_at`, …) are added alongside them.
4. **Embed and store** (`ingest_chunks`): embeds a metadata header plus the chunk text with `text-embedding-3-small`, then upserts into Qdrant with the full payload. Point IDs are a `uuid5` of `chunk_id`, so re-ingesting overwrites rather than duplicates. Keyword indexes on the metadata fields let `retrieve()` filter on them.
5. **Retrieve** (`retrieve`): runs the chosen mode (see the table above) and returns the top k = 3 chunks. `filters` can narrow any mode by metadata, e.g. `{"source": "02_speckled_band.txt"}`.
6. **Answer** (`answer`): `gpt-4o-mini` answers from the retrieved excerpts only, reusing their exact wording and citing story title and section. Returns the answer, citations and end-to-end latency.
7. **Validate** (`validate_against`): for each question, checks that the expected story is cited (source-match) and counts how many expected facts appear in the answer as substrings.

To compare modes, change the default `mode` argument of `answer()` and run `validate` again.

## Results

Full tables and analysis are in [mp2_reflection.md](mp2_reflection.md). Summary of the logged runs ([mp2_validation.txt](mp2_validation.txt)):

| Run | Mode | Source-match | Facts matched | Avg latency |
|---|---|:-:|:-:|:-:|
| Round 1 | `simple` | 5/5 | 14/22 | ~3.3 s |
| Round 2 | `refined` | 5/5 | 17/22 | ~7.5 s |
| Round 3 | `refined` (repeat) | 5/5 | 17/22 | ~8.1 s |

- Section-aware chunking kept source-match at 5/5 in every run and made citations specific.
- Sub-queries helped most on multi-part questions: q5 rose from 2/5 to 4/5 facts in round 2.
- The round 3 run meant for `hybrid` ran `refined`, because `validate` uses `answer()`'s default mode. Hybrid is implemented but not yet validated.
- Individual questions moved by a fact between the two refined runs with no code changes, so small gaps between modes are within noise.

## Configuration

Constants at the top of `src/mp2_rag.py`:

| Setting | Default |
|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` (1536 dims) |
| `CHAT_MODEL` | `gpt-4o-mini` (metadata, query refinement and answers) |
| `MAX_CHUNK_SIZE` | 1000 characters |
| `COLLECTION_NAME` | `Sherlock_collection` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

Note: the first `hybrid` question builds the BM25 index and loads the cross-encoder, which downloads about 80 MB the first time. That question is much slower than the rest.
