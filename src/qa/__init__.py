"""Grounded QA layer over a scored grant application.

Three modes, each reusing an existing store / artefact:
  - single_doc : retrieve the current application's chunks (in-memory store)
  - scoring    : explain the AI assessment (belief_state + scored features + evidence)
  - corpus     : compare across the labelled ES corpus (committee/admin only)

An intent router (one LLM call, with query rewrite) picks the mode and a
standalone retrieval query. Answers are grounded + cited (chunk text + page).
"""
