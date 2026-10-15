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
subgraph "multiple repeats"
    embedding(Embedding)
    mlp(MLP)
    norm1(Normalization)
    add1(Add)
    attention(Multi-head attention)
    norm2(Normalization)
    add2(Add)
end
output(Output)

input --> embedding
embedding --> add1
embedding --> mlp
mlp --> norm1
norm1 --> add1
add1 --> add2
add1 --> attention
attention --> norm2
norm2 --> add2
add2 --> output
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