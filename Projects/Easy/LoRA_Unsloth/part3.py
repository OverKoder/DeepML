from datasets import Dataset


def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    instruction_response_list = [
        {
            "instruction": "How do I plant a fruit tree?",
            "response": "First you need to choose the right location for the tree. Make sure it's an area with lots of sunlight, good drainage, not directly near any buildings, and not too close to other trees. Then purchase a healthy fruit tree from your local nursery and prepare the soil for planting. Dig a hole for the tree that is twice as wide as the root ball, and as deep as the length of the root ball. Place the tree in the hole and fill the soil around it, making sure the soil is firmly packed. Finally, water the tree thoroughly and place mulch or straw around the base of the tree.",
        },
        {
            "instruction": "What should I look for when buying a used car?",
            "response": "Engine, transmission, brakes, and any other mechanical components, and ensure that the car is within your budget. Additionally, if the car has been in an accident, get a full report of its history. Finally, be sure to take a test drive of the car and make sure that it handles as you expect.",
        },
        {
            "instruction": "How many planets are in our solar system?",
            "response": "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
        },
    ]

    return instruction_response_list


def format_instruction_example(instruction_response):
    return f"### Instruction:\n{instruction_response['instruction']}\n\n### Response:\n{instruction_response['response']}"


def format_all_examples(instruction_response_list):
    return [
        format_instruction_example(instruct_response)
        for instruct_response in instruction_response_list
    ]


def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    return Dataset.from_dict({"text": texts})


def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # Call the tokenizer on text and return its input_ids as a plain list
    return tokenizer(text, return_tensors="pt").input_ids.squeeze(0).tolist()


def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    return len(input_ids)
