---
okf_version: "1.0"
type: Concept
title: Hydra Head Settlement
timestamp: "2026-07-03T18:35:00Z"
status: draft
source_docs:
  - hydra-settlement-test-fixture-4edb3c3c
---

# Hydra Head Settlement

Hydra settlement treats a Hydra Head as a synchronous settlement layer over
Cardano-style eUTXO state: once all participants sign a snapshot, that state
is final among them even before it touches the main chain. Safety relies on
unanimity plus a contestation period during Close rather than an
honest-majority assumption.

Back to the [knowledge base index](/index.md).
