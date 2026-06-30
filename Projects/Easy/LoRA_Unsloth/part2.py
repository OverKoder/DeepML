from unsloth import FastLanguageModel


def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        target_modules=target_modules
        if target_modules is not None
        else get_lora_target_modules(),
    )
    return model


def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    count = 0
    for param in model.parameters():
        if param.requires_grad:
            count += param.numel()
    return count


def trainable_fraction(trainable_count, total_count):
    return trainable_count / total_count
