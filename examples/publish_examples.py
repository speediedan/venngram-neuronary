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
"""Translate the example recipes in this directory and publish them as venngram documents.

Every ``<model dir>/*.json`` here is a recipe written for these examples in neuronary's format (never one of neuronary's
own). Each is read through :class:`venngram_neuronary.NeuronaryFormat`, with the model revision resolved on the Hub at
translation time, written with :func:`venngram.sources.write_translation`, and published with
:func:`venngram.store.publish` to the model directory's target (``TARGETS``), or only written locally with ``--out``.

.. code-block:: bash

    python examples/publish_examples.py --out /tmp/neuronary-examples     # translate only
    python examples/publish_examples.py                                   # translate and publish (needs a Hub token)
"""
import argparse
import json
import tempfile
from pathlib import Path

from venngram.sources import ReadContext, write_translation
from venngram.store import publish

from venngram_neuronary import NeuronaryFormat

HERE = Path(__file__).parent
#: Example model directory -> (store, ref) its translations are published to.
TARGETS = {"qwen3.5-4b": ("hub-model", "speediedan/venngram-qwen3.5-4b/neuronary")}


def hub_revision(model_id: str) -> str:
    from huggingface_hub import HfApi

    sha = HfApi().model_info(model_id).sha
    if not sha:
        raise RuntimeError(f"could not resolve a revision for {model_id}")
    return sha


def translate(model_dir: Path, out: Path) -> list[tuple[str, str]]:
    ctx = ReadContext(resolve_revision=hub_revision)
    documents: list[tuple[str, str]] = []
    for source in sorted(model_dir.glob("*.json")):
        doc = json.loads(source.read_text(encoding="utf-8"))
        documents += write_translation(NeuronaryFormat().read(doc, ctx), out)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="write the translations here and publish nothing")
    args = parser.parse_args()
    for name, (store, ref) in TARGETS.items():
        if args.out is not None:
            documents = translate(HERE / name, args.out / name)
            print(f"{name}: wrote {[p for _, p in documents]} under {args.out / name}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            documents = translate(HERE / name, Path(tmp))
            manifest, revision = publish(store, ref, tmp, documents=documents,
                                         message=f"publish neuronary-format examples ({NeuronaryFormat.version})")
            print(f"{name}: published {[e.path for e in manifest.documents]} to {store}:{ref} at {revision}")


if __name__ == "__main__":
    main()
