from unsloth.trainer import TrainingArguments, SFTTrainer
import torch


def build_training_arguments(output_dir="./sft_out", max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    # TODO: build TrainingArguments with batch size 1, given max_steps, given lr, bf16 or fp16.
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        logging_steps=1,
        optim="adamw_8bit",
    )
    return args


def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=max_seq_length,
        packing=True,
    )
    return trainer


def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    output = trainer.train()
    return output.training_loss
