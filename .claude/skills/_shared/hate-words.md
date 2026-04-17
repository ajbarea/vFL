# AI-slop hate-word glossary

Canonical cross-skill list. Referenced by `/deslop`, `/reslop`, `/docsync`.
Update this file, not the individual skills.

Each section is a **candidate generator**, not a verdict — a hit only starts
the conversation. Always filter against the Keep rules of the skill that's
using the list. "Robust" inside a user-facing error message is fine;
"robust implementation" in a docstring is slop.

## Marketing / hype padding

- `robust`, `comprehensive`, `elegant(ly)?`
- `powerful`, `blazing(ly)?`, `lightning[- ]fast`, `battle[- ]tested`
- `production[- ]ready`, `enterprise[- ]grade`, `industry[- ]standard`
- `seamless(ly)?`, `effortlessly`, `with ease`, `painlessly`
- `simply`, `just` (as a filler, not a verb), `out[- ]of[- ]the[- ]box`
- `state[- ]of[- ]the[- ]art`, `cutting[- ]edge`, `next[- ]generation`
- `future[- ]proof` (unless the code provably is)

## Temporal / versioning rot

- `best[- ]practice` (e.g. "based on April 2026 best practice")
- `as of 20\d\d`, `latest version`, `current(ly)? recommended`
- `modern(ly)?`, `up[- ]to[- ]date`

## Planning / task-context rot

- `PHASE ?\d`, `Phase [0-9A-Z]`, `Step \d of \d`, `Part [IVX]+`
- `TODO\((?:copilot|gpt|gemini|claude|cursor)`
- `added for (?:issue|ticket|pr) #\d+`, `fix for the .* bug`

## Self-referential AI framing

- `AI (?:assistant|debugger|debug)`
- `LLM`, `for (?:model|AI) consumption`
- `designed so an? (?:AI|assistant|model)`
- `AI-DEBUG`, `AI HINTS?`

## Verbose verbs (replace with simpler)

- `leverag(es?|ing)` → use
- `utiliz(es?|ing)` → use
- `facilitat(es?|ing)` → let / allow
- `encapsulat(es?|ing)` → hold / wrap (when overused)

## Hedging / filler docstring prose

- `it's important to note`, `it is worth noting`, `please note that`
- `in summary`, `under the hood`, `at a high level`
- `we might want to`, `in some cases this could`, `potentially`
- `needless to say`, `as mentioned earlier`

## Narrative WHAT-comment patterns

Flagged manually — greppable but high false-positive rate:

- `# Now we ...`, `# Then we ...`, `# Finally ...`
- `# This (?:function|method|class) ...` (often restates the signature)
- `# Return the ...` when the return type is already declared
- Bullet lists in docstrings that only repeat argument types
