# Pretraining transformers from scratch

## Architecture

File architecture:

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

The transformer model:

```mermaid
flowchart TD

input(Input)
subgraph "x4 repeats"
    embedding(Embedding)
    add(Add)
    subgraph "x4 repeats "
        mlp(MLP)
        attention(Multi-head attention)
    end
end
output(Output)

input --> embedding
embedding --> add
embedding --> mlp
mlp --> attention
attention --> add
add --> output
```

The multi-head attention block:
```mermaid
flowchart TD

subgraph "n x (Multiple heads) parallel"
input(Input embeddings)
key(Key vector)
query(Query vector)
value(Value vector)
attention(Attention scores)
mask_attention(Causally masked attention scores)
softmax_attention(Softmax attention scores)
weighted_value(Weighted value vector)
end
output(Output)

input --> key
input --> query
input --> value
key --> attention
query --> attention
attention --> mask_attention
mask_attention --> softmax_attention
softmax_attention --> weighted_value
value --> weighted_value
weighted_value --> output
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