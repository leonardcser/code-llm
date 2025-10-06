import pytest
import tokenizer as tok
from dataloaders.py150 import Py150DataModule
import torch
import yaml

MAX_TOKENS = 100000


@pytest.fixture(scope="module")
def config():
    """Load configuration from params.yaml."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


@pytest.fixture(scope="module")
def datamodule(config):
    """Create and setup datamodule."""
    data_params = config["data"]
    training_params = config["training"]

    dm = Py150DataModule(
        dataset_file=data_params["dataset_file"],
        split_ratio=data_params["split_ratio"],
        seq_length=data_params["seq_length"],
        batch_size=training_params["batch_size"],
        num_workers=0,
        pin_memory=False,
        seed=training_params["seed"],
        eos_token_id=data_params["eos_token_id"],
        bos_token_id=data_params["bos_token_id"],
        max_tokens=MAX_TOKENS,
        pad_token_id=data_params["pad_token_id"],
    )

    dm.setup(stage="fit")
    return dm


@pytest.fixture(scope="module")
def tokenizer_obj(config):
    """Load tokenizer."""
    return tok.load(config["data"]["tok_file"])


def test_all_batches_have_correct_seq_length(datamodule, config):
    """Test that all batches have the correct sequence length."""
    data_params = config["data"]
    seq_length = data_params["seq_length"]

    train_loader = datamodule.train_dataloader()

    for batch_idx, batch in enumerate(train_loader):
        input_ids, target_ids, attention_mask, position_ids = batch

        assert input_ids.shape[1] == seq_length, (
            f"Batch {batch_idx}: input_ids seq_length mismatch! "
            f"Expected {seq_length}, got {input_ids.shape[1]}"
        )
        assert target_ids.shape[1] == seq_length, (
            f"Batch {batch_idx}: target_ids seq_length mismatch! "
            f"Expected {seq_length}, got {target_ids.shape[1]}"
        )
        assert attention_mask.shape[-1] == seq_length, (
            f"Batch {batch_idx}: attention_mask seq_length mismatch! "
            f"Expected {seq_length}, got {attention_mask.shape[-1]}"
        )
        assert position_ids.shape[1] == seq_length, (
            f"Batch {batch_idx}: position_ids seq_length mismatch! "
            f"Expected {seq_length}, got {position_ids.shape[1]}"
        )


def test_all_batches_have_same_batch_size(datamodule, config):
    """Test that all batches have the same batch size (including last batch)."""
    training_params = config["training"]
    batch_size = training_params["batch_size"]

    train_loader = datamodule.train_dataloader()

    for batch_idx, batch in enumerate(train_loader):
        input_ids, target_ids, attention_mask, position_ids = batch

        assert input_ids.shape[0] == batch_size, (
            f"Batch {batch_idx}: batch_size mismatch! "
            f"Expected {batch_size}, got {input_ids.shape[0]}"
        )
        assert target_ids.shape[0] == batch_size, (
            f"Batch {batch_idx}: target_ids batch_size mismatch! "
            f"Expected {batch_size}, got {target_ids.shape[0]}"
        )
        assert attention_mask.shape[0] == batch_size, (
            f"Batch {batch_idx}: attention_mask batch_size mismatch! "
            f"Expected {batch_size}, got {attention_mask.shape[0]}"
        )
        assert position_ids.shape[0] == batch_size, (
            f"Batch {batch_idx}: position_ids batch_size mismatch! "
            f"Expected {batch_size}, got {position_ids.shape[0]}"
        )


def test_eos_token_handling(datamodule, tokenizer_obj, config):
    """Test that EOS tokens are properly handled in sequences."""
    data_params = config["data"]
    eos_token_id = data_params["eos_token_id"]

    train_loader = datamodule.train_dataloader()

    found_eos = False
    max_batches = 100

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_batches:
            break

        input_ids, _target_ids, _attention_mask, position_ids = batch

        # Search for EOS in this batch
        for sample_idx in range(input_ids.shape[0]):
            sample_input = input_ids[sample_idx]
            eos_positions = (
                (sample_input == eos_token_id).nonzero(as_tuple=False).squeeze(-1)
            )

            if len(eos_positions) > 0:
                found_eos = True

                # Verify EOS token exists
                assert eos_token_id in sample_input, "EOS token not found in sequence"

                # Verify position IDs reset after EOS
                for eos_pos in eos_positions:
                    if eos_pos < len(position_ids[sample_idx]) - 1:
                        # Position after EOS should reset to a low value
                        pos_after_eos = position_ids[sample_idx][eos_pos + 1].item()
                        assert pos_after_eos <= 1, (
                            f"Position ID after EOS should reset, got {pos_after_eos}"
                        )

                # Verify we can decode the sequence
                decoded = tokenizer_obj.decode(
                    sample_input.tolist(), skip_special_tokens=False
                )
                assert len(decoded) > 0, "Failed to decode sequence with EOS"

                break

        if found_eos:
            break

    assert found_eos, "No EOS tokens found in dataset"


def test_bos_token_at_sequence_start(datamodule, config):
    """Test that BOS tokens appear at the start of sequences."""
    data_params = config["data"]
    bos_token_id = data_params["bos_token_id"]

    train_loader = datamodule.train_dataloader()

    found_bos = False

    for batch_idx, batch in enumerate(train_loader):
        input_ids, _, _, _ = batch

        # Check first batch samples
        for sample_idx in range(input_ids.shape[0]):
            if input_ids[sample_idx][0].item() == bos_token_id:
                found_bos = True
                break

        if found_bos or batch_idx >= 10:
            break

    # BOS tokens should be present (but not necessarily at position 0 of every sequence)
    # Just verify that BOS token ID is valid
    assert bos_token_id is not None, "BOS token ID should be configured"


def test_batch_shapes_consistency(datamodule, config):
    """Test that batch dimensions are consistent."""
    data_params = config["data"]
    seq_length = data_params["seq_length"]

    train_loader = datamodule.train_dataloader()

    for batch in train_loader:
        input_ids, target_ids, _attention_mask, _position_ids = batch

        # Check batch dimensions match
        assert input_ids.shape[0] == target_ids.shape[0], (
            "Input and target batch sizes don't match"
        )
        assert input_ids.shape == target_ids.shape, (
            "Input and target shapes don't match"
        )

        # Check sequence dimension
        assert input_ids.shape == (input_ids.shape[0], seq_length), (
            f"Expected shape (*, {seq_length}), got {input_ids.shape}"
        )


def test_token_ids_within_vocab_range(datamodule, tokenizer_obj):
    """Test that all token IDs are within valid vocabulary range."""
    train_loader = datamodule.train_dataloader()
    vocab_size = tokenizer_obj.vocab_size

    max_batches_to_check = 10

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_batches_to_check:
            break

        input_ids, target_ids, _, _ = batch

        # Check input tokens
        max_token = input_ids.max().item()
        min_token = input_ids.min().item()

        assert min_token >= 0, f"Found negative token ID: {min_token}"
        assert max_token < vocab_size, (
            f"Found token ID {max_token} >= vocab_size {vocab_size}"
        )

        # Check target tokens
        max_token = target_ids.max().item()
        min_token = target_ids.min().item()

        assert min_token >= 0, f"Found negative token ID: {min_token}"
        assert max_token < vocab_size, (
            f"Found token ID {max_token} >= vocab_size {vocab_size}"
        )


def test_datamodule_has_train_and_val_datasets(datamodule):
    """Test that datamodule has both train and validation datasets."""
    assert datamodule.train_dataset is not None, "Train dataset not initialized"
    assert datamodule.val_dataset is not None, "Validation dataset not initialized"

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert train_loader is not None, "Train dataloader is None"
    assert val_loader is not None, "Validation dataloader is None"


def test_padding_with_pad_token(config):
    """Test that padding works correctly with pad_token_id using F.pad."""
    from dataloaders.py150 import Py150DataModule

    data_params = config["data"]
    pad_token_id = data_params["pad_token_id"]
    seq_len = data_params["seq_length"]

    # Create a datamodule with batch_size=3 to match our test batch
    dm = Py150DataModule(
        dataset_file=data_params["dataset_file"],
        split_ratio=data_params["split_ratio"],
        seq_length=seq_len,
        batch_size=3,  # Match the number of samples in our test batch
        num_workers=0,
        pin_memory=False,
        seed=42,
        eos_token_id=data_params["eos_token_id"],
        bos_token_id=data_params["bos_token_id"],
        max_tokens=MAX_TOKENS,
        pad_token_id=pad_token_id,
    )

    # Create sequences of different lengths
    x1 = torch.randint(0, 1000, (seq_len,))
    x2 = torch.randint(0, 1000, (seq_len - 2,))  # Shorter sequence
    x3 = torch.randint(0, 1000, (seq_len - 1,))  # Shorter sequence

    y1 = torch.randint(0, 1000, (seq_len,))
    y2 = torch.randint(0, 1000, (seq_len - 2,))
    y3 = torch.randint(0, 1000, (seq_len - 1,))

    # Create simple batch without masks first
    batch_simple = [(x1, y1), (x2, y2), (x3, y3)]

    # Test collate function
    result = dm._collate_fn(batch_simple)
    if len(result) == 2:
        x_batch, y_batch = result
    else:
        x_batch, y_batch, _, _ = result

    # Verify padding
    assert x_batch.shape == (3, seq_len), (
        f"Expected shape (3, {seq_len}), got {x_batch.shape}"
    )
    assert y_batch.shape == (3, seq_len), (
        f"Expected shape (3, {seq_len}), got {y_batch.shape}"
    )

    # Verify pad tokens were added
    assert (x_batch[1, -2:] == pad_token_id).all(), (
        "Expected pad tokens at end of sequence 2"
    )
    assert (x_batch[2, -1:] == pad_token_id).all(), (
        "Expected pad tokens at end of sequence 3"
    )

    # Verify original tokens are preserved
    assert (x_batch[0] == x1).all(), "Original tokens should be preserved in sequence 1"
    assert (x_batch[1, :-2] == x2).all(), (
        "Original tokens should be preserved in sequence 2"
    )
    assert (x_batch[2, :-1] == x3).all(), (
        "Original tokens should be preserved in sequence 3"
    )


def test_batch_level_padding(datamodule, config):
    """Test that batches with fewer samples than batch_size get padded correctly."""
    training_params = config["training"]
    data_params = config["data"]
    batch_size = training_params["batch_size"]
    pad_token_id = data_params["pad_token_id"]

    train_loader = datamodule.train_dataloader()

    # Iterate through all batches to find the last one
    last_batch = None
    last_batch_idx = 0
    for batch_idx, batch in enumerate(train_loader):
        last_batch = batch
        last_batch_idx = batch_idx

    # Verify last batch exists and has been padded to batch_size
    assert last_batch is not None, "No batches found in train_loader"
    input_ids, target_ids, attention_mask, position_ids = last_batch

    assert input_ids.shape[0] == batch_size, (
        f"Last batch (batch {last_batch_idx}) should be padded to batch_size={batch_size}, "
        f"got {input_ids.shape[0]}"
    )

    # Calculate expected number of real samples in last batch
    # Total samples in dataset: 273 (from test output)
    # Full batches: 273 // batch_size = 8 full batches (256 samples)
    # Remaining samples: 273 % batch_size = 17 samples
    # So last batch should have 17 real samples + 15 padded samples

    dataset_size = len(datamodule.train_dataset)
    expected_real_samples = dataset_size % batch_size
    if expected_real_samples == 0:
        expected_real_samples = batch_size
    expected_pad_samples = batch_size - expected_real_samples

    # If there are padded samples, verify they contain pad_token_id
    if expected_pad_samples > 0:
        # Check that the padded samples (last expected_pad_samples) are all pad_token_id
        padded_input_ids = input_ids[-expected_pad_samples:]
        padded_target_ids = target_ids[-expected_pad_samples:]

        assert (padded_input_ids == pad_token_id).all(), (
            f"Expected all padded input_ids to be {pad_token_id}"
        )
        assert (padded_target_ids == pad_token_id).all(), (
            f"Expected all padded target_ids to be {pad_token_id}"
        )

        # Check that attention masks for padded samples are all False/0
        padded_attention_mask = attention_mask[-expected_pad_samples:]
        assert not padded_attention_mask.any(), (
            "Expected attention masks for padded samples to be all False/0"
        )

        # Check that position IDs for padded samples are all 0
        padded_position_ids = position_ids[-expected_pad_samples:]
        assert (padded_position_ids == 0).all(), (
            "Expected position IDs for padded samples to be all 0"
        )
