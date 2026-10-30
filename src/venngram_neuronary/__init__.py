# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A venngram format adapter for `neuronary <https://github.com/lyramakesmusic/neuronary>`_ recipes.

neuronary publishes neuron-steering recipes as JSON (``schema/recipe.schema.json``): a model id, a list of
``{layer, neuron, dose}``, an optional ``method`` (``op``, ``site``, ``normalization``, ``apply_to``), sampling
settings, example generations and descriptive fields. :class:`NeuronaryFormat` reads one into a venngram checkpoint
book (the neurons as per-layer units at the input of ``mlp.out``) and a schema 2 venngram recipe (one stage, one op,
the doses as one dose or a per-unit table). It is found by venngram through the ``venngram.formats`` entry point.

Lossless or refused: what venngram cannot carry yet (an MoE ``expert``, an ``apply_to`` with no venngram token scope)
is refused by name, unless the read context is lenient, when it is listed in the translation's ``unexpressed`` and kept
under the recipe's ``metadata``. ``prompt_tokens`` and ``generated_tokens`` become venngram's ``prompt`` and
``generated`` scopes. neuronary pins no model revision, so the read context's
resolver supplies one and the recipe records ``revision_source: resolved-at-import``.
"""
import hashlib
import json
from typing import Any

from venngram.errors import Refusal
from venngram.sources import Provenance, ReadContext, SourceError, Translation

__version__ = "0.1.0.dev0"

#: neuronary sites -> venngram activation points.
SITES = {"mlp.down_proj.input": "mlp.out.hook_in"}

#: neuronary ``apply_to`` -> venngram token scopes.
TOKENS = {"all_tokens": "all", "prompt_tokens": "prompt", "generated_tokens": "generated"}

#: neuronary ``normalization`` -> a venngram value transform (``None``: raw doses).
NORMALIZATION = {"raw": None, "layer_residual_median_over_down_column_l2": "layer_residual_median_over_down_column_l2"}

_DESCRIPTIVE = ("id", "description", "tags", "author", "date", "aliases", "source", "notes", "conditional")
_DEFAULT_METHOD = {"op": "add", "site": "mlp.down_proj.input",
                   "normalization": "layer_residual_median_over_down_column_l2", "apply_to": "all_tokens"}


class NeuronaryFormat:
    """Reads neuronary recipes (see the module docstring)."""

    name = "neuronary"
    version = __version__
    kinds = ("address_book", "recipe")
    unexpressed = ("moe_experts",)

    def detect(self, doc: Any) -> bool:
        return isinstance(doc, dict) and isinstance(doc.get("neurons"), list) and isinstance(doc.get("model"), str) \
            and isinstance(doc.get("id"), str) and all(isinstance(n, dict) and "neuron" in n for n in doc["neurons"])

    def read(self, doc: Any, ctx: ReadContext) -> Translation:
        if not self.detect(doc):
            raise SourceError("neuronary: not a neuronary recipe (needs `id`, `model` and `neurons`).")
        name, model_id = doc["id"], doc["model"]
        method = {**_DEFAULT_METHOD, **(doc.get("method") or {})}
        refusals: list[Refusal] = []
        neurons = []
        for n in doc["neurons"]:
            if "expert" in n:
                refusals.append(Refusal(f"neuron L{n['layer']}E{n['expert']}N{n['neuron']} names an MoE expert, which "
                                        "venngram does not address yet"))
            else:
                neurons.append(n)
        if method["site"] not in SITES:
            raise SourceError(f"neuronary: {name}: unknown site {method['site']!r}; known: {sorted(SITES)}.")
        if method["normalization"] not in NORMALIZATION:
            raise SourceError(f"neuronary: {name}: unknown normalization {method['normalization']!r}.")
        tokens = TOKENS.get(method["apply_to"])
        if tokens is None:
            refusals.append(Refusal(f"apply_to `{method['apply_to']}` has no venngram token scope yet"))
        if refusals and not ctx.lenient:
            raise SourceError(f"neuronary: {name}: {refusals[0]} (read with a lenient context to keep it as metadata).")
        if not neurons:
            raise SourceError(f"neuronary: {name}: no neuron venngram can address.")
        if ctx.resolve_revision is None:
            raise SourceError(f"neuronary: {name}: neuronary pins no revision; the read context needs a "
                              "`resolve_revision`.")
        revision = ctx.resolve_revision(model_id)
        site = SITES[method["site"]]
        units: dict[int, list[int]] = {}
        for n in neurons:
            units.setdefault(int(n["layer"]), []).append(int(n["neuron"]))
        book = {"schema_version": 1, "name": name, "portability": "checkpoint",
                "applies_to": {"models": [{"id": model_id, "revision": revision}]},
                "sets": {name: {"point": site, "layers": sorted(units),
                                "units": {k: sorted(set(v)) for k, v in sorted(units.items())}}}}
        if isinstance(doc.get("description"), str):
            book["description"] = doc["description"]
        doses = {float(n["dose"]) for n in neurons}
        value: dict[str, Any]
        if len(doses) == 1:
            value = {"dose": doses.pop()}
        else:
            table: dict[int, dict[int, float]] = {}
            for n in neurons:
                table.setdefault(int(n["layer"]), {})[int(n["neuron"])] = float(n["dose"])
            value = {"doses": table}
        statistic = NORMALIZATION[method["normalization"]]
        if statistic is not None:
            value["transform"] = [{"statistic": {"name": statistic}}]
        metadata: dict[str, Any] = {"imported_from": "neuronary", **{k: doc[k] for k in _DESCRIPTIVE if k in doc}}
        if refusals:
            metadata["unexpressed"] = [str(r) for r in refusals]
        recipe = {"schema_version": 2, "kind": "recipe", "name": doc.get("name", name),
                  "model": {"id": model_id, "revision": revision, "revision_source": "resolved-at-import"},
                  "book": f"../books/{name}.yaml",
                  "stages": [{"site": site, "tokens": tokens or "all",
                              "ops": [{"op": "scale" if method["op"] == "mul" else "add", "target": {"set": name},
                                       "value": value}]}],
                  "sampling": dict(doc.get("sampling") or {}),
                  "evidence": {"examples": list(doc.get("examples") or [])}, "metadata": metadata}
        digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
        return Translation({f"books/{name}.yaml": ("address_book", book), f"recipes/{name}.yaml": ("recipe", recipe)},
                           Provenance("format", self.name, self.version, digest, "resolved-at-import"),
                           unexpressed=tuple(refusals))


__all__ = ["NeuronaryFormat", "NORMALIZATION", "SITES", "TOKENS"]
