# Fixture sources

The four real fixtures are agent instruction files copied verbatim from public
repositories. All credit to the original authors; each file remains under its
source repository's license.

- `codex_agents.md` — AGENTS.md from https://github.com/openai/codex
- `kiali_agents.md` — AGENTS.md from https://github.com/kiali/kiali
- `temporal_agents.md` — AGENTS.md from https://github.com/temporalio/temporal
- `freerouting_agents.md` — AGENTS.md from https://github.com/freerouting/freerouting
  (held out: rules and checkers were written before the pipeline ever processed
  this file)

`dev_claude.md` and `ops_meridian.md` are synthetic files written for this
benchmark.

The `.atoms.json` files are cached MenteDB parses of the fixtures, and the
`.rules.json` files anchor each tested rule to verbatim text from the fixture.
