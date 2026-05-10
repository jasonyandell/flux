# AGENTS.md — flux Wiki Schema

This file tells any LLM agent how to read, write, and extend this wiki.

## What this wiki is

Three layers:

- **Raw sources** — the repo's code and any digests in `sources/`. Read-only except when adding digests.
- **The wiki** — markdown under `wiki/` minus `sources/`. LLM-owned, mutable, kept current.
- **The schema** — this file.

Default mode for any agent in this repo: **consult the wiki first, then read code.** Update the wiki as a side effect of work that changes what is true. Do not batch wiki updates.

## Core philosophy

The wiki tracks **the frontier**: what the project understands to be true now.

- Pages evolve in place.
- Do not preserve stale claims as history unless the history itself matters.
- `log.md` carries timeline; normal pages state the current truth.
- A useful answer that required synthesis should be filed back into the wiki before the session ends.

The loop:

```txt
query wiki → read sources/code → change project → update wiki
```

## Directory layout

```txt
wiki/
├── AGENTS.md          ← this file
├── index.md           ← catalog and route map
├── log.md             ← chronological append-only wiki updates
├── entities/          ← named things: flux, the game itself
├── topics/            ← concepts: galcon-like, force vectors
├── decisions/         ← explicit design choices
├── sources/           ← compact source digests
└── questions/open.md  ← unresolved questions
```

Do not create pages speculatively. Create a page when work needs it.

## Page conventions

Every page starts with YAML frontmatter:

```yaml
---
title: Human Readable Title
kind: index | log | entity | topic | decision | source | question
first_seen: bootstrap | <commit-shortsha>
last_updated: bootstrap | <commit-shortsha>
status: active | retired | superseded
---
```

If no commit is available, use `bootstrap` or `workspace`.

Body rules:

- 3rd person, declarative.
- Terse. Wiki pages are reference, not essays.
- Use bare backlinks on first mention in a section: `[[continuous-flow-model]]`, not `[[decisions/continuous-flow-model]]`.
- Cite source digests inline when useful: `([bootstrap](../sources/bootstrap.md))`.
- Use `##` and `###` headings. No top-level `#`; the title is frontmatter.

Filenames: lowercase, hyphen-separated, no spaces or underscores.

## Operations

### Query — every session

1. Start at `index.md` or the relevant entity page.
2. Follow backlinks to decisions/topics.
3. Read code only after the wiki gives the map.
4. Cite wiki paths when answering.
5. If the answer required new synthesis, update the wiki.

### Update — whenever truth changes

Update the wiki in the same session when:

- A game rule changes.
- A new source file or module appears.
- A decision is retired or superseded.
- A query reveals a missing page or stale claim.

Update steps:

1. Decide which pages move.
2. Rewrite pages to state the new frontier.
3. Add or update a source digest if the change is meaningful.
4. Update `index.md` for new/renamed pages.
5. Append one entry to `log.md`.
6. Sweep for broken backlinks and stale claims.

## Log format

`log.md` entries use:

```md
## [YYYY-MM-DD | <rev> | <subject>]

**Touched pages:** [[flux]] [[continuous-flow-model]]
**Added:** ...
**Updated:** ...
**Retired:** ...
**Questions opened:** ...
```

## Backlinks, not categories

Organization emerges from backlinks and short route maps. Keep it simple enough that the wiki compounds without becoming a second codebase.
