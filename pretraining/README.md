# Pretraining transformers from scratch

## Architecture

```mermaid
flowchart TD
main(main.py - argument parsing, calling training loop)
model(lib/model.py - the transformer model)
training(lib/training.py - the pretraining loop)
data(lib/data.py - download pretraining datasets, create input/output pairs, tokenize)
generate(lib/generate.py - generate new text)

main --> model
main --> training
main --> data
main --> generate
```

## Running

Run the code:

```bash
uv run main.py
```

Checks before committing:

```bash
uvx ty check && uvx ruff check --fix && uv run pytest
```