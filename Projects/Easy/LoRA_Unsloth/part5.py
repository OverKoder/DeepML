from unsloth import FastLanguageModel


def switch_to_inference_mode(model):
    """Switch the LoRA-tuned model into Unsloth's fast inference mode and return it."""
    FastLanguageModel.for_inference(model)
    return model


def build_chat_prompt(tokenizer, instruction):
    """Return a chat-template prompt string ready for assistant generation."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy-generate a reply for `prompt` and return the decoded text."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[-1]

    response = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)[
        0
    ][input_length:]
    return tokenizer.decode(response, skip_special_tokens=True)
