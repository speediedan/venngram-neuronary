# venngram-neuronary

A [venngram](https://github.com/speediedan/venngram) format adapter for
[neuronary](https://github.com/lyramakesmusic/neuronary) neuron-steering recipes. It reads a neuronary recipe (JSON, per
neuronary's `schema/recipe.schema.json`) into a venngram checkpoint address book and a venngram recipe (schema 2), which
venngram can then publish, compose and check like any other documents.

```python
import json

from venngram.sources import format_for, ReadContext, write_translation

doc = json.load(open("some-recipe.json"))
translation = format_for(doc).read(doc, ReadContext(resolve_revision=my_revision_resolver))
documents = write_translation(translation, "out/")   # books/<id>.yaml, recipes/<id>.yaml
```

Installing the package registers the format with venngram through the `venngram.formats` entry point, so
`format_for(doc)` finds it.

- **Neurons** become per-layer units at the input of `mlp.out` (neuronary's `mlp.down_proj.input`).
- **Doses** become one dose when they agree, else a per-unit table. `mul` becomes venngram's `scale`, and neuronary's
  default normalization becomes a named `statistic` transform.
- **Examples** become the recipe's evidence, and the descriptive fields go under `metadata`.
- **Revisions.** neuronary pins no model revision, so the read context's resolver supplies one and the recipe records
  `revision_source: resolved-at-import`.
- **What venngram cannot carry yet** is refused by name: an MoE `expert`, and an `apply_to` of `prompt_tokens` or
  `generated_tokens`. A lenient read context instead records it under `metadata`.

This package contains code only. It does not include or redistribute neuronary's recipes; the test fixtures are written
for the tests. Experimental, like venngram itself.

Apache-2.0.
