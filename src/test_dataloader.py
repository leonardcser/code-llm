from tokenizer import Tokenizer
from dataloaders.data_loader import get_dataloaders
import yaml


with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

data_params = params["data"]
model_params = params["model"]
training_params = params["training"]
other_params = params["other"]

# Set random seeds for reproducibility
seed = training_params.get("seed", 42)
eos_token_id = data_params.get("eos_token_id")

train_loader, val_loader = get_dataloaders(
    data_params["train_file"],
    data_params["val_file"],
    seq_length=data_params["seq_length"],
    batch_size=training_params["batch_size"],
    num_workers=0,
    seed=seed,
    eos_token_id=eos_token_id,
)

# Initialize tokenizer for decoding
tokenizer = Tokenizer("out/tokenize/tok.bin")

for batch in train_loader:
    # Unpack batch
    input_ids, target_ids, attention_mask, position_ids = batch

    print("=" * 80)
    print("BATCH PREVIEW - First Element")
    print("=" * 80)

    # Get first element of batch
    first_input = input_ids[0]
    first_target = target_ids[0]
    first_mask = attention_mask[0]
    first_pos = position_ids[0]

    print("\nBatch shapes:")
    print(f"  Input IDs:       {input_ids.shape}")
    print(f"  Target IDs:      {target_ids.shape}")
    print(f"  Attention Mask:  {attention_mask.shape}")
    print(f"  Position IDs:    {position_ids.shape}")

    print("\n--- First Element (index 0) ---")
    print(f"\nInput tokens (first 20):  {first_input[:20].tolist()}")
    print(f"Target tokens (first 20): {first_target[:20].tolist()}")

    print("\nDecoded Input Text:")
    print(f"  {tokenizer.decode(first_input.tolist())[:200]}...")

    print("\nDecoded Target Text:")
    print(f"  {tokenizer.decode(first_target.tolist())[:200]}...")

    print(f"\nAttention Mask (first 20): {first_mask[:20].tolist()}")
    print(f"Position IDs (first 20):   {first_pos[:20].tolist()}")

    # Find EOS tokens if any
    eos_positions = (first_input == eos_token_id).nonzero(as_tuple=False).squeeze(-1)
    if len(eos_positions) > 0:
        print(f"\nEOS token positions in sequence: {eos_positions.tolist()}")
    else:
        print("\nNo EOS tokens found in this sequence")

    print("=" * 80)
    exit()
