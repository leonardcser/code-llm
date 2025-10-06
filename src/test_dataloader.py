from tokenizer import Tokenizer
from dataloaders.py150_dataloader import TokenDataset
import torch
from torch.utils.data import DataLoader
import yaml


with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

data_params = params["data"]
model_params = params["model"]
training_params = params["training"]

# Set random seeds for reproducibility
seed = training_params["seed"]
eos_token_id = data_params["eos_token_id"]
bos_token_id = data_params["bos_token_id"]

# Create datasets
train_dataset = TokenDataset(
    data_params["train_file"],
    seq_length=data_params["seq_length"],
    eos_token_id=eos_token_id,
    bos_token_id=bos_token_id,
)
val_dataset = TokenDataset(
    data_params["val_file"],
    seq_length=data_params["seq_length"],
    eos_token_id=eos_token_id,
    bos_token_id=bos_token_id,
)

# Create dataloaders
generator = None
if seed is not None:
    generator = torch.Generator()
    generator.manual_seed(seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=training_params["batch_size"],
    shuffle=True,
    num_workers=0,
    generator=generator,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=training_params["batch_size"],
    shuffle=False,
    num_workers=0,
)

# Initialize tokenizer for decoding
tokenizer = Tokenizer("out/tokenize/tok.bin")

print(f"Searching for a sample with EOS token (id={eos_token_id})...")
print(f"BOS token ID: {bos_token_id}\n")

max_batches = 100  # Limit search to avoid infinite loops
found_eos = False

for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= max_batches:
        print(f"Searched {max_batches} batches without finding EOS token")
        break

    # Unpack batch
    input_ids, target_ids, attention_mask, position_ids = batch

    # Search for EOS in this batch
    for sample_idx in range(input_ids.shape[0]):
        sample_input = input_ids[sample_idx]
        eos_positions = (
            (sample_input == eos_token_id).nonzero(as_tuple=False).squeeze(-1)
        )

        if len(eos_positions) > 0:
            # Found a sample with EOS!
            found_eos = True
            first_input = sample_input
            first_target = target_ids[sample_idx]
            first_mask = attention_mask[sample_idx]
            first_pos = position_ids[sample_idx]

            print("=" * 80)
            print(f"SAMPLE WITH EOS - Batch {batch_idx}, Sample {sample_idx}")
            print("=" * 80)

            print("\nBatch shapes:")
            print(f"  Input IDs:       {input_ids.shape}")
            print(f"  Target IDs:      {target_ids.shape}")
            print(f"  Attention Mask:  {attention_mask.shape}")
            print(f"  Position IDs:    {position_ids.shape}")

            print(f"\n--- Sample {sample_idx} (contains EOS) ---")
            print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
            print(f"Tokenizer BOS token ID: {tokenizer.bos_token_id}")
            print(f"Tokenizer EOS token ID: {tokenizer.eos_token_id}")
            print(
                f"\nFirst token in input (should be BOS={bos_token_id}): {first_input[0].item()}"
            )
            print(f"Input tokens (first 30):  {first_input[:30].tolist()}")
            print(f"Target tokens (first 30): {first_target[:30].tolist()}")
            print(f"Input tokens (last 30):   {first_input[-30:].tolist()}")
            print(f"Target tokens (last 30):  {first_target[-30:].tolist()}")

            print("\nFull decoded input (with special tokens):")
            full_decoded_with = tokenizer.decode(
                first_input.tolist(), skip_special_tokens=False
            )
            print(full_decoded_with)

            print("\n" + "=" * 80)
            print("\nFull decoded input (skip special tokens):")
            full_decoded_skip = tokenizer.decode(
                first_input.tolist(), skip_special_tokens=True
            )
            print(full_decoded_skip)

            print(f"\nEOS token positions in sequence: {eos_positions.tolist()}")

            print(f"\nPosition IDs (all {len(first_pos)} values):")
            print(f"  {first_pos.tolist()}")

            # Check for invalid token IDs
            max_token = first_input.max().item()
            min_token = first_input.min().item()
            print(f"\nToken ID range in input: [{min_token}, {max_token}]")
            if max_token >= tokenizer.vocab_size:
                print(
                    f"WARNING: Found token IDs >= vocab_size ({tokenizer.vocab_size})"
                )
                invalid_tokens = (first_input >= tokenizer.vocab_size).sum().item()
                print(f"  Number of invalid tokens: {invalid_tokens}")

            print("=" * 80)
            break

    if found_eos:
        break

if not found_eos:
    print("No samples with EOS token found in the dataset")

exit()
