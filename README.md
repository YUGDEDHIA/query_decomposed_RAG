# query_decomposed_RAG

A from-scratch implementation of **query-decomposition agentic RAG**, running
entirely on local models via [Ollama](https://ollama.com).

## What this is

Plain RAG embeds the whole user question and does one vector search. That
falls apart on compound questions ("compare X and Y's approach to Z, and
mention when W happened") — a single embedding of the whole question sits in
a vector-space no-man's-land and retrieves mediocre chunks for every part of
it instead of great chunks for any one part.

This project adds a planning step before retrieval: an LLM call decomposes
the question into independent sub-questions, each gets its own clean
retrieval pass and its own grounded mini-answer, and a final synthesis step
combines everything into one coherent answer to the original question.

```
[docs] --chunk+embed--> [vector store]

user query
   │
   ▼
┌─────────┐   sub-q 1 ──retrieve──> chunks ──┐
│ Planner │   sub-q 2 ──retrieve──> chunks ──┼──> ┌────────────┐
│ (LLM)   │   sub-q 3 ──retrieve──> chunks ──┘    │ Synthesizer│──> final answer
└─────────┘                                        │   (LLM)    │
                                                    └────────────┘
```

## Models (via Ollama)

```
ollama pull qwen2.5:14b-instruct   # planner + synthesizer + per-sub-question answering
ollama pull nomic-embed-text        # embeddings
```

(`qwen2.5:32b-instruct` is also worth trying on 48GB+ RAM — noticeably more
reliable structured-output/planning behavior at the cost of slower
inference.)

## Project layout

```
src/            implementation (built up commit by commit — see git log)
tests/          unit tests for the pieces that don't need a live LLM call
                (chunking, vector store) plus manual smoke-test scripts for
                the pieces that do (planner, retrieval, synthesis)
data/raw/       test corpus (tracked in git — small text docs)
data/index/     persisted embeddings (gitignored — rebuilt by the ingestion
                script)
```

## Status

Scaffolding only so far. The pipeline is built incrementally — see commit
history for the build order (ingestion → retrieval → planner →
per-sub-question answering → synthesizer → end-to-end pipeline).
