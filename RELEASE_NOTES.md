# Release notes

Copyright 2026 Zale Pan.

## First experimental source candidate

This is the first public source candidate of the five-package workspace: Core runtime, Fake
adapter, Web adapter, Scripted planner and LLM planner.

### What is included

* the Core runtime, contract builder, step verifier, task oracle and recovery engine;
* the Fake and Web environment adapters;
* the Scripted planner and the offline LLM planner with its retrieval and evidence path;
* offline examples; the whitelisted source exporter is an internal release tool and is not part of the public file set;
* the Apache License 2.0 license text.

### Historical verified results (earlier rounds)

* offline runtime, retrieval and evidence demonstrations complete through the real Core loop;
* the deterministic offline test selection passes from the exported candidate;
* the exporter copies exactly a manifest-declared whitelist and refuses traversal, links, protected
  prefixes, duplicate targets and hash or length mismatches.

### Not run / unknown

* a live provider call is **not_run**: no API request was made and no credential was read;
* the behaviour of a provider that hangs during cleanup remains **unknown**;
* browser reference runs need a separately installed Chromium build and were not part of the offline
  validation of this candidate;
* no production reliability, capacity, any-website coverage or cross-platform guarantee is claimed.

### Stability

The interfaces may change without notice. Treat this candidate as experimental.

### Round-006 historical results

* the four outer command classes (ruff, collect, public-offline, public-installation) and the
  thirteen installation-internal commands were recorded with real timings in round-006;
* the public offline selection passed 3809 tests with one skip and one deselect there;
* the two Web local runs and the exporter controls are historical references, re-used, not re-run.

### Round-007

* seven public documents were corrected against the round-007 checklist; no product code, test,
  build, archive or environment was changed or re-run in this round;
* nothing here claims the public candidate has been independently verified yet.
