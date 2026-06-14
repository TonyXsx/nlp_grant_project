"""Hybrid-retrieval layer for the grant scorer.

Reuses the vendored swxy DeepDOC retriever (``Dealer``) over two stores:
  - an ephemeral in-memory store for the current application (evidence), and
  - a persistent Elasticsearch corpus of labelled applications (few-shot).
"""
