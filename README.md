# k3-kv

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/k3-kv/card.svg" alt="k3-kv Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-kv/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-kv/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-kv/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-kv/benchmark.svg" alt="Benchmark"/></a>
</p>

---

Context memory holder daemon and shared memory arena management for the SpacemiT K3 AI platform.

## Overview
`k3-kv` is a minimal, policy-free daemon process holding per-model context memory arenas:
- **Policy-Free Memory Holder**: Allocates and locks (`mlock`) shared-memory arenas within operator `RLIMIT_MEMLOCK` bounds.
- **FD Passing via SCM_RIGHTS**: Passes arena file descriptors to `k3-manager` and `k3-worker` processes over Unix Domain Sockets.
- **Fault Survival**: Survives worker abnormal termination and supervisor reconnects, retaining active KV context blocks.

## Architecture & Specifications
For master architecture and detailed component specifications, see the parent [model-runner](https://github.com/LMA1980/model-runner) repository:
- Master Plan: [PLAN.md](https://github.com/LMA1980/model-runner/blob/main/PLAN.md)
- Specs: [specs/memory/kv-holder.md](https://github.com/LMA1980/model-runner/blob/main/specs/memory/kv-holder.md)
