## Hydra Settlement: A Synchronous Layer-2 for eUTXO Chains

Working Paper, Test Fixture Edition

## 1. Introduction

Hydra is a family of Layer-2 protocols designed to scale Cardano-style eUTXO ledgers by allowing a small set of participants to run an isolated, high-throughput state channel called a Hydra Head. Transactions inside a Head settle instantly among participants and are only reconciled with the main chain when the Head is closed.

## 2. Architecture

## 2.1 Head Lifecycle

A Hydra Head moves through four phases: Init, Open, Close, and Fanout. During Init, participants commit UTXOs from the main chain. During Open, participants exchange signed snapshots off-chain. Close posts the latest snapshot on-chain, and Fanout distributes the final UTXO set back to the main ledger.

## 2.2 Settlement Guarantees

Because every state transition inside the Head requires unanimous signatures from all participants, Hydra can be used as a synchronous settlement layer over Cardano-style eUTXO state: once all parties sign a snapshot, that state is final among them even before it touches the main chain.

## 3. Throughput Comparison

The table below compares theoretical transaction throughput across configurations tested in the reference implementation.

| Configuration      | Participants   |   TPS | Finality   |
|--------------------|----------------|-------|------------|
| Baseline L1        | -              |   250 | ~20s       |
| Hydra Head (small) | 3              | 1,000 | <1s        |
| Hydra Head (large) | 10             |   800 | <1s        |

## 4. Threat Model

Hydra Heads assume an honest majority is not required; instead, safety relies on unanimity and a contestation period during Close. If a participant posts a stale snapshot, any other participant can contest it on-chain within the contestation window by presenting a newer, validly signed snapshot.

## 5. Open Questions

It remains an open question how Hydra Heads compose with external delegated-signing standards such as the Open Wallet Standard (OWS), particularly when an autonomous agent - rather than a human participant - holds one of the Head's signing keys.

Figure 1: Head Lifecycle (placeholder)