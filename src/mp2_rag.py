"""MP2 · Mini-RAG — Starter Template
====================================

You'll build a complete RAG pipeline over the Sherlock Holmes corpus in this
file. Fill in every TODO. The reference solution is ~250 lines, but yours can
be shorter or longer — what matters is that it works end-to-end.

Pipeline you're building:
    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations

Run sequence (once you've filled in the TODOs):
    pip install -r requirements.txt
    source .env                 # exports your OpenAI + Qdrant credentials
    python mp2_rag.py ingest    # builds the collection (run once)
    python mp2_rag.py ask       # interactive Q&A loop
    python mp2_rag.py validate  # runs against data/predefined_questions.jsonl

Tip: get the CORE pipeline working FIRST (Steps 1-7 below), THEN come back to
polish and add your 3 questions. Don't try to perfect each step before moving
on — you'll learn more from a rough end-to-end loop than a polished half.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

# ─── Configuration ──────────────────────────────────────────────────────

PROJECT_DIR       = Path(__file__).resolve().parent.parent  # src/ lives one level below
CORPUS_DIR        = PROJECT_DIR / "corpus"
DATA_DIR          = PROJECT_DIR / "data"
COLLECTION_NAME   = "Sherlock_collection"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
EMBEDDING_BATCH_SIZE = 100  # texts per embeddings API call
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP     = 80    # characters
MAX_CHUNK_SIZE    = 1000  # characters — hard upper bound for any chunk
# Payload fields indexed in Qdrant and accepted by retrieve(filters=...).
FILTERABLE_FIELDS = ("source", "section", "characters", "places", "key_items", "occurrence_date")
# Strategies accepted by retrieve(mode=...).
RETRIEVAL_MODES   = ("simple", "refined", "hybrid")

openai = OpenAI()
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ.get("QDRANT_API_KEY"),
)


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in corpus_dir into a {source, title, text} dict.

    Files are read in sorted order so ingestion is deterministic. The title is
    the first non-empty line, falling back to the filename stem.
    """
    docs: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        # First non-empty line is the story title; stem covers empty files.
        title = next((line.strip() for line in text.splitlines() if line.strip()), path.stem)
        docs.append({"source": path.name, "title": title, "text": text})
    return docs


# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into {source, title, section, text} chunks.

    Structure-aware: paragraphs are grouped under the section header that
    precedes them, then each section is split with _split_recursive so no chunk
    exceeds MAX_CHUNK_SIZE or crosses a section boundary.
    """
    # Pass 1: group paragraphs by section. Text before the first header
    # belongs to "Introduction"; the title line itself is skipped.
    sections: list[tuple[str, list[str]]] = [("Introduction", [])]
    for block in re.split(r"\n\s*\n", doc["text"]):
        block = block.strip()
        if not block or block == doc["title"]:
            continue
        if _is_section_header(block):
            sections.append((block, []))
        else:
            sections[-1][1].append(block)

    # Pass 2: recursively split each section into size-bounded chunks,
    # tagging each with its section for citations.
    chunks: list[dict[str, Any]] = []
    for section, paragraphs in sections:
        if not paragraphs:
            continue
        for text in _split_recursive("\n\n".join(paragraphs), MAX_CHUNK_SIZE):
            chunks.append({
                "chunk_id": f"{Path(doc['source']).stem}#{len(chunks)}",
                "source": doc["source"],
                "title": doc["title"],
                "section": section,
                "text": text,
            })
    return chunks


# Coarsest to finest: paragraphs, lines, sentences, words, characters.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _is_section_header(block: str) -> bool:
    """A header is a short, single-line block without terminal punctuation."""
    return "\n" not in block and len(block) <= 80 and not block.endswith((".", "!", "?", ":", ";", ",", '"', "'", "”", "’"))


def _split_recursive(text: str, max_size: int, separators: list[str] = _SEPARATORS) -> list[str]:
    """Split text on the coarsest separator, recursing into finer separators
    only for pieces that are still too long, then greedily re-pack the pieces
    into chunks of at most max_size characters."""
    text = text.strip()
    if len(text) <= max_size:
        return [text] if text else []

    sep, finer = separators[0], separators[1:]
    if sep == "":
        return [text[i:i + max_size] for i in range(0, len(text), max_size)]

    parts = text.split(sep)
    # Keep the separator attached so re-packed chunks read naturally.
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]

    units: list[str] = []
    for piece in pieces:
        if len(piece.strip()) > max_size:
            # Restore the piece's trailing whitespace on its last sub-chunk so
            # it joins cleanly with whatever unit follows.
            subs = _split_recursive(piece, max_size, finer)
            subs[-1] += piece[len(piece.rstrip()):]
            units.extend(subs)
        elif piece.strip():
            units.append(piece)

    chunks: list[str] = []
    buf = ""
    for unit in units:
        if buf and len((buf + unit).strip()) > max_size:
            chunks.append(buf.strip())
            buf = ""
        buf += unit
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs). Texts are
    sent in batches to stay under the API's per-request input limit.
    """
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        response = openai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # Sort by index so vectors line up with inputs regardless of response order.
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return vectors


# ─── Step 3b: Enrich chunk metadata ─────────────────────────────────────

METADATA_PROMPT = """You extract metadata from an excerpt of a Sherlock Holmes story.
Return a JSON object with exactly these keys:
  "characters":      list of people who appear or are named, using the fullest name given in the excerpt
  "places":          list of locations where events happen or that are named (streets, towns, buildings, rooms)
  "occurrence_date": the date or period when the events happen (e.g. "April 1883"), or null if not stated
  "occurrence_time": the time of day when the events happen (e.g. "early morning", "midnight"), or null if not stated
  "key_items":       list of objects that matter to the plot (e.g. weapons, clues, valuables)
Use only what the excerpt says. Use empty lists or null when nothing applies."""

_LIST_FIELDS = ("characters", "places", "key_items")
_TEXT_FIELDS = ("occurrence_date", "occurrence_time")


def extract_story_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Ask the chat model for the story fields in METADATA_PROMPT.

    Output is normalised so every field is present with the right type, even
    if the model omits a key or returns an unexpected value.
    """
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": METADATA_PROMPT},
            {"role": "user", "content": f"Story: {chunk['title']}\nSection: {chunk['section']}\n\n{chunk['text']}"},
        ],
    )
    raw = json.loads(response.choices[0].message.content or "{}")
    meta: dict[str, Any] = {}
    for field in _LIST_FIELDS:
        values = raw.get(field) or []
        meta[field] = [str(v).strip() for v in values if str(v).strip()] if isinstance(values, list) else []
    for field in _TEXT_FIELDS:
        value = raw.get(field)
        meta[field] = str(value).strip() if value else None
    return meta


def enrich_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Build the full payload for one chunk, following the WHAMAgent schema:
    provenance fields (chunk_id, source, doc_type, version, ingested_at) plus
    the story fields extracted by the LLM."""
    return {
        **chunk,
        "doc_type":    Path(chunk["source"]).suffix.lstrip(".").lower(),
        "language":    "en",
        "version":     "v1",
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "char_count":  len(chunk["text"]),
        **extract_story_metadata(chunk),
    }


def embedding_text(chunk: dict[str, Any]) -> str:
    """Text actually embedded for a chunk: a short metadata header followed by
    the chunk body, so the vector also matches queries that name the story,
    section, people or places the body only refers to indirectly."""
    header = [f"Story: {chunk['title']}", f"Section: {chunk['section']}"]
    if chunk.get("characters"):
        header.append("Characters: " + ", ".join(chunk["characters"]))
    if chunk.get("places"):
        header.append("Places: " + ", ".join(chunk["places"]))
    when = ", ".join(v for v in (chunk.get("occurrence_date"), chunk.get("occurrence_time")) if v)
    if when:
        header.append(f"When: {when}")
    if chunk.get("key_items"):
        header.append("Key items: " + ", ".join(chunk["key_items"]))
    return "\n".join(header) + "\n\n" + chunk["text"]


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection.

    recreate_collection drops any existing collection with the same name, so
    every ingest starts from an empty index sized for EMBEDDING_DIM vectors
    compared by cosine similarity. Keyword indexes on the metadata fields let
    retrieve() filter on them.
    """
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    for field in FILTERABLE_FIELDS:
        qdrant.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=PayloadSchemaType.KEYWORD)


# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Enrich every chunk with metadata, embed it, and upsert into Qdrant.

    The vector is built from embedding_text (metadata header + body); the
    payload stores the clean body plus every metadata field for filtering and
    citations.
    """
    # Metadata extraction is one LLM call per chunk, so run them in parallel.
    with ThreadPoolExecutor(max_workers=8) as pool:
        enriched = list(pool.map(enrich_metadata, chunks))

    vectors = embed_texts([embedding_text(c) for c in enriched])

    # uuid5 of chunk_id gives stable IDs, so re-ingesting overwrites instead of duplicating.
    points = [
        PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])), vector=vec, payload=c)
        for c, vec in zip(enriched, vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  {len(points)} chunks upserted into '{COLLECTION_NAME}'")


# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(
    query: str,
    k: int = 3,
    filters: dict[str, str | list[str]] | None = None,
    mode: str = "refined",
) -> list[dict[str, Any]]:
    """Retrieve chunks for a query using one of three strategies.

    mode="simple":  a single top-k cosine search.
    mode="refined": (the default) refine_query() targets a story and splits
        the query into metadata-tagged sub-queries. Each sub-query runs two
        searches inside that story: a plain vector search, and one narrowed to
        chunks tagged with ANY of the sub-query's metadata values. The two
        rankings are combined by reciprocal rank fusion, the top k kept, and
        the per-sub-query results interleaved so every part of the query is
        covered. If refinement fails, this falls back to the simple search.
    mode="hybrid":  dense + BM25 retrieval fused by RRF, then reranked by a
        cross-encoder; see _hybrid_rerank_retrieve().

    filters optionally narrows every search on payload metadata, e.g.
    {"source": "02_speckled_band.txt"} or {"characters": ["Irene Adler"]}.
    A list matches chunks whose field contains any of the values. An explicit
    "source" filter overrides the story the refiner picks.

    Returns the chunk payloads with a "score" key added for citations and,
    when refined, a "sub_queries" key listing the sub-queries each chunk
    was retrieved for.
    """
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; choose from {RETRIEVAL_MODES}")

    # Branch 3: hybrid dense + BM25, fused by RRF and reranked.
    if mode == "hybrid":
        return _hybrid_rerank_retrieve(query, k=k, filters=filters)

    # Branch 2: refined sub-queries with metadata filters.
    refined = None
    if mode == "refined":
        try:
            refined = refine_query(query)
        except Exception as exc:  # refinement is an optimisation; never block retrieval
            print(f"  (query refinement failed: {exc}; using plain retrieval)", file=sys.stderr)

    # Branch 1: simple top-k cosine search (also the refined-mode fallback).
    if refined is None:
        return _search(embed_texts([query])[0], k, _build_filter(filters))

    must = _filter_conditions(filters)
    if refined["source"] and not (filters and "source" in filters):
        must.append(FieldCondition(key="source", match=MatchValue(value=refined["source"])))
    subs = refined["sub_queries"]
    vectors = embed_texts([sub["query"] for sub in subs])

    def rank_sub(sub: dict[str, Any], vector: list[float]) -> list[dict[str, Any]]:
        rankings = [_search(vector, k * 2, Filter(must=must) if must else None)]
        if sub["metadata"]:
            tagged: list[Condition] = [FieldCondition(key=f, match=MatchAny(any=v)) for f, v in sub["metadata"].items()]
            rankings.append(_search(vector, k * 2, Filter(must=must or None, should=tagged)))
        fused: dict[str, float] = {}
        chunks: dict[str, dict[str, Any]] = {}
        for ranking in rankings:
            for rank, chunk in enumerate(ranking):
                fused[chunk["chunk_id"]] = fused.get(chunk["chunk_id"], 0.0) + 1 / (_RRF_K + rank)
                chunks.setdefault(chunk["chunk_id"], chunk)
        return [chunks[cid] for cid in sorted(fused, key=lambda cid: fused[cid], reverse=True)[:k]]

    with ThreadPoolExecutor(max_workers=len(subs)) as pool:
        per_sub = list(pool.map(rank_sub, subs, vectors))
    # Interleave round-robin so each sub-query's best chunk comes first.
    merged: dict[str, dict[str, Any]] = {}
    for rank in range(k):
        for sub, ranked in zip(subs, per_sub):
            if rank < len(ranked):
                chunk = merged.setdefault(ranked[rank]["chunk_id"], {**ranked[rank], "sub_queries": []})
                chunk["sub_queries"].append(sub["query"])
    return list(merged.values())


def _search(vector: list[float], k: int, query_filter: Filter | None) -> list[dict[str, Any]]:
    """Top-k payloads (plus "score") for an already-embedded query."""
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    ).points
    return [{**(hit.payload or {}), "score": round(hit.score, 4)} for hit in hits]


def _build_filter(filters: dict[str, str | list[str]] | None) -> Filter | None:
    """Turn {field: value | [values]} into a Qdrant Filter that ANDs the fields."""
    conditions = _filter_conditions(filters)
    return Filter(must=conditions) if conditions else None


def _filter_conditions(filters: dict[str, str | list[str]] | None) -> list[Condition]:
    """One exact-match condition per {field: value | [values]} entry."""
    conditions: list[Condition] = []
    for field, value in (filters or {}).items():
        if field not in FILTERABLE_FIELDS:
            raise ValueError(f"Cannot filter on {field!r}; choose from {FILTERABLE_FIELDS}")
        match = MatchAny(any=value) if isinstance(value, list) else MatchValue(value=value)
        conditions.append(FieldCondition(key=field, match=match))
    return conditions


# ─── Step 6a: Hybrid retrieval + cross-encoder rerank ───────────────────

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_hybrid_cache: dict[str, Any] = {}


def _hybrid_rerank_retrieve(
    query: str,
    k: int = 3,
    filters: dict[str, str | list[str]] | None = None,
    n_candidates: int = 20,
) -> list[dict[str, Any]]:
    """Dense + BM25 retrieval fused by RRF, then reranked by a cross-encoder.

    1. Dense:  top n_candidates by cosine similarity from Qdrant.
    2. Sparse: top n_candidates by BM25 over every chunk's embedding_text, so
       exact names, places and items in the query are matched literally.
    3. Fuse:   reciprocal rank fusion, score = sum(1 / (_RRF_K + rank)).
    4. Rerank: the cross-encoder scores each (query, chunk text) pair of the
       fused pool, and the top k are kept.

    The BM25 index (built from a scroll of the collection) and the
    cross-encoder (~80MB download on first use) are cached after the first
    call. filters works as in retrieve() and applies to both retrievers.

    Returns chunk payloads with "score" set to the rerank score and
    "rrf_score" set to the fusion score.
    """
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder

    def tokenize(text: str) -> list[str]:
        # Keeps hyphenated / slashed tokens (e.g. "st-clair") whole.
        return re.findall(r"[a-z0-9][a-z0-9\-/_]*", text.lower())

    if not _hybrid_cache:
        corpus: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = qdrant.scroll(
                COLLECTION_NAME, limit=256, offset=offset, with_payload=True, with_vectors=False,
            )
            corpus.extend(point.payload or {} for point in points)
            if offset is None:
                break
        _hybrid_cache["corpus"] = corpus
        _hybrid_cache["bm25"] = BM25Okapi([tokenize(embedding_text(c)) for c in corpus])
        _hybrid_cache["reranker"] = CrossEncoder(RERANKER_MODEL)
    corpus, bm25, reranker = _hybrid_cache["corpus"], _hybrid_cache["bm25"], _hybrid_cache["reranker"]

    def matches(chunk: dict[str, Any]) -> bool:
        """Python-side equivalent of _build_filter for the BM25 results."""
        for field, value in (filters or {}).items():
            wanted = set(value) if isinstance(value, list) else {value}
            have = chunk.get(field)
            if not wanted & (set(have) if isinstance(have, list) else {have}):
                return False
        return True

    dense = _search(embed_texts([query])[0], n_candidates, _build_filter(filters))
    bm25_scores = bm25.get_scores(tokenize(query))
    sparse = [
        chunk for score, chunk in sorted(zip(bm25_scores, corpus), key=lambda p: p[0], reverse=True)
        if score > 0 and matches(chunk)
    ][:n_candidates]

    fused: dict[str, float] = {}
    chunks: dict[str, dict[str, Any]] = {}
    for ranking in (dense, sparse):
        for rank, chunk in enumerate(ranking, start=1):
            fused[chunk["chunk_id"]] = fused.get(chunk["chunk_id"], 0.0) + 1 / (_RRF_K + rank)
            chunks.setdefault(chunk["chunk_id"], chunk)
    pool = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:n_candidates]
    if not pool:
        return []

    rerank_scores = reranker.predict([(query, chunks[cid]["text"]) for cid in pool])
    ranked = sorted(zip(pool, rerank_scores), key=lambda p: p[1], reverse=True)[:k]
    return [
        {**chunks[cid], "score": round(float(score), 4), "rrf_score": round(fused[cid], 4)}
        for cid, score in ranked
    ]


# ─── Step 6b: Refine the query against the metadata ─────────────────────

# Metadata fields a refined sub-query can be matched on.
_MATCH_FIELDS = ("section", "characters", "places", "key_items")
_RRF_K = 60  # standard reciprocal-rank-fusion damping constant

REFINE_PROMPT = """You turn a question about a collection of Sherlock Holmes stories into
precise search queries. You are given a catalog of the metadata stored for each
story: its sections and the characters, places and key items tagged in it.

Return a JSON object with exactly these keys:
  "source":      the file name of the story the question is about, or null if unclear
  "sub_queries": one entry per distinct part of the question (1 to 4 entries), each with
      "query":      a standalone search query for that part, rewritten to include the
                    catalog terms (names, places, items, section titles) it is likely about
      "section":    up to 2 section titles most likely to hold the answer to this part
      "characters": catalog characters this part is about or likely to be named in its answer
      "places":     catalog places this part is about or likely to be named in its answer
      "key_items":  catalog key items likely to appear in the answer to this part; include the
                    most specific names (e.g. "swamp adder" as well as "snake")
Copy catalog values exactly as written. Use empty lists when nothing fits."""


def metadata_catalog() -> dict[str, dict[str, Any]]:
    """Distinct metadata values per source, read from the ingested collection.

    Cached after the first call so refinement costs no extra Qdrant round
    trips per question.
    """
    if _catalog_cache:
        return _catalog_cache
    offset = None
    while True:
        points, offset = qdrant.scroll(
            COLLECTION_NAME, limit=256, offset=offset, with_payload=True, with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            entry = _catalog_cache.setdefault(
                payload["source"], {"title": payload.get("title", ""), **{f: {} for f in _MATCH_FIELDS}},
            )
            for field in _MATCH_FIELDS:
                values = payload.get(field) or []
                # dict keys keep first-seen order and drop duplicates.
                entry[field].update(dict.fromkeys(values if isinstance(values, list) else [values]))
        if offset is None:
            return _catalog_cache


_catalog_cache: dict[str, dict[str, Any]] = {}


def refine_query(question: str) -> dict[str, Any]:
    """Ask the chat model to target a story and split the question into
    metadata-tagged sub-queries.

    Every returned value is checked against the catalog, so later filters only
    use values that actually exist in the collection.
    """
    catalog = metadata_catalog()
    catalog_text = "\n\n".join(
        f"{source} ({entry['title']})\n"
        + "\n".join(f"  {field}: {'; '.join(entry[field])}" for field in _MATCH_FIELDS)
        for source, entry in catalog.items()
    )
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": REFINE_PROMPT},
            {"role": "user", "content": f"Catalog:\n\n{catalog_text}\n\nQuestion: {question}"},
        ],
    )
    raw = json.loads(response.choices[0].message.content or "{}")

    source = raw.get("source") if raw.get("source") in catalog else None
    # Without a story, validate against the union of every story's values.
    known = {
        field: set(catalog[source][field]) if source
        else {v for entry in catalog.values() for v in entry[field]}
        for field in _MATCH_FIELDS
    }
    sub_queries = []
    for sub in (raw.get("sub_queries") or [])[:4]:
        if not isinstance(sub, dict) or not str(sub.get("query") or "").strip():
            continue
        matched: dict[str, list[str]] = {}
        for field in _MATCH_FIELDS:
            values = sub.get(field) or []
            values = values if isinstance(values, list) else [values]
            if kept := [v for v in values if v in known[field]]:
                matched[field] = kept
        sub_queries.append({"query": str(sub["query"]).strip(), "metadata": matched})
    return {"source": source, "sub_queries": sub_queries or [{"query": question, "metadata": {}}]}


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Answer every part of the
question listed under "Parts to answer", giving every relevant detail the
excerpts state for it: who, what, where, when, how, the evidence and the
motive. Reuse the excerpts' exact wording for names, places, objects, dates and
key phrases rather than paraphrasing, and give people's full names when any
excerpt states them. Cite the source (story title + section) in
your answer."""


def answer(question: str, k: int = 3, mode: str = "refined") -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result.

    mode selects the retrieval strategy (see retrieve()).

    Returns {question, answer, citations, latency_ms}. Each citation carries
    source, title, section, chunk_id and score of a retrieved chunk, and
    latency_ms covers the whole pipeline (embedding, search and generation).
    """
    start = time.perf_counter()
    chunks = retrieve(question, k=k, mode=mode)
    # The refined sub-queries the chunks were retrieved for, in first-seen order.
    parts = list(dict.fromkeys(q for c in chunks for q in c.get("sub_queries", []))) or [question]

    # Label each excerpt so the model can cite story title + section.
    context = "\n\n".join(
        f"[Source: {c['title']} — {c['section']}]\n{c['text']}" for c in chunks
    )
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Question: {question}\n\nParts to answer:\n"
                + "\n".join(f"- {part}" for part in parts)
                + f"\n\nExcerpts:\n\n{context}"
            )},
        ],
    )

    citations = [
        {key: c[key] for key in ("source", "title", "section", "chunk_id", "score")}
        for c in chunks
    ]
    return {
        "question": question,
        "answer": (response.choices[0].message.content or "").strip(),
        "citations": citations,
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }


# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print (f"      A: {result['answer']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        facts_missing = [fact for fact in q.get("expected_facts", []) if fact.lower() not in ans_lower]
        print(f"      Facts missing: {', '.join(facts_missing) if facts_missing else 'none'}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
