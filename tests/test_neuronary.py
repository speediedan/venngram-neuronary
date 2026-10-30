"""Reading neuronary-format recipes. The fixtures are written for these tests, in neuronary's schema."""
import copy

import pytest

from venngram.manifest import build_manifest
from venngram.recipe import Recipe
from venngram.sources import format_for, ReadContext, SourceError, write_translation
from venngram.store import load, publish
from venngram_neuronary import NeuronaryFormat

RECIPE = {
    "id": "two-neurons", "name": "Two neurons", "description": "a made-up recipe in neuronary's shape",
    "model": "Qwen/Qwen3.5-4B", "tags": ["test"], "author": {"handle": "venngram-tests"}, "date": "2026-10-30",
    "neurons": [{"layer": 13, "neuron": 7956, "dose": -1.0}, {"layer": 14, "neuron": 8949, "dose": 1.0}],
    "sampling": {"repetition_penalty": 1.1},
    "examples": [{"prompt": "hi", "output": "hello"}],
}
CTX = ReadContext(resolve_revision=lambda model_id: "abc123")


def test_a_recipe_translates_publishes_and_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("VENNGRAM_CACHE", str(tmp_path / "cache"))
    translation = NeuronaryFormat().read(RECIPE, CTX)
    documents = write_translation(translation, tmp_path / "src")
    assert {e.path for e in build_manifest(tmp_path / "src", documents).documents} == {
        "books/two-neurons.yaml", "recipes/two-neurons.yaml"}
    publish("local", str(tmp_path / "pub"), tmp_path / "src", documents=documents)
    recipe = load(f"local:{tmp_path / 'pub'}:recipes/two-neurons.yaml")
    assert isinstance(recipe, Recipe) and recipe.revision_source == "resolved-at-import"
    (stage,) = recipe.stages
    assert stage.site == "mlp.out.hook_in" and stage.tokens.kind == "all"
    (op,) = stage.ops
    assert op.op == "add" and op.unit_doses(recipe.book) == {(13, 7956): -1.0, (14, 8949): 1.0}
    assert [t.name for t in op.value.transform] == ["statistic"]  # neuronary's default normalization
    assert recipe.evidence["examples"] == RECIPE["examples"] and recipe.metadata["imported_from"] == "neuronary"
    assert translation.provenance.source == "neuronary" and len(translation.provenance.input_sha256) == 64
    assert load(f"local:{tmp_path / 'pub'}:books/two-neurons.yaml").description == RECIPE["description"]


def test_one_dose_mul_and_raw_doses():
    doc = {**RECIPE, "neurons": [{"layer": 15, "neuron": n, "dose": 0.8} for n in (4041, 17)],
           "method": {"op": "mul", "normalization": "raw"}}
    recipe = NeuronaryFormat().read(doc, CTX).documents["recipes/two-neurons.yaml"][1]
    assert recipe["stages"][0]["ops"] == [{"op": "scale", "target": {"set": "two-neurons"}, "value": {"dose": 0.8}}]


@pytest.mark.parametrize("change, match", [
    (lambda d: d["neurons"].append({"layer": 1, "expert": 3, "neuron": 2, "dose": 1.0}), "MoE expert"),
    (lambda d: d.update(method={"apply_to": "first_token"}), "no venngram token scope"),
])
def test_what_venngram_cannot_carry_is_refused_unless_lenient(change, match):
    doc = copy.deepcopy(RECIPE)
    change(doc)
    with pytest.raises(SourceError, match=match):
        NeuronaryFormat().read(doc, CTX)
    lenient = NeuronaryFormat().read(doc, ReadContext(resolve_revision=lambda m: "abc123", lenient=True))
    assert len(lenient.unexpressed) == 1
    assert lenient.documents["recipes/two-neurons.yaml"][1]["metadata"]["unexpressed"]


def test_a_revision_resolver_is_required_and_the_format_is_discovered_by_entry_point():
    with pytest.raises(SourceError, match="needs a `resolve_revision`"):
        NeuronaryFormat().read(RECIPE, ReadContext())
    assert format_for(RECIPE).name == "neuronary"  # through the installed `venngram.formats` entry point


@pytest.mark.parametrize("apply_to, scope", [("prompt_tokens", "prompt"), ("generated_tokens", "generated")])
def test_prompt_and_generated_tokens_become_venngram_scopes(apply_to, scope):
    doc = {**RECIPE, "method": {"apply_to": apply_to}}
    translation = NeuronaryFormat().read(doc, CTX)
    recipe = Recipe.from_mapping(translation.documents["recipes/two-neurons.yaml"][1] | {
        "book": translation.documents["books/two-neurons.yaml"][1]})
    assert recipe.stages[0].tokens.kind == scope and not translation.unexpressed
