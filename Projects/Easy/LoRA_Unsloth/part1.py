from unsloth import FastLanguageModel
from bitsandbytes.nn import Linear4bit


def load_base_model_and_tokenizer(
    model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit", max_seq_length=256
):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    return model, tokenizer


def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    count = 0
    for param in model.parameters():
        count += param.numel()
    return count


def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # Run through every submodules, but only check leaf modules.
    return any(
        [
            isinstance(elem, Linear4bit)
            for elem in model.modules()
            if len(list(elem.children())) == 0
        ]
    )


def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
