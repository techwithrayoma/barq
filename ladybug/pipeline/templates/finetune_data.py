from string import Template

SYSTEM_FINETUNE = Template(
    "\n".join([
        "You are an NLP data parser specialized in text classification.",
        "Your task is to extract the intent of a YouTube comment.",
        "Only output JSON with the predicted intent.",
        "Do not add explanations."
    ])
)

INSTRUCTION_FINETUNE = Template(
    "## YouTube Comment:\n$comment\n\n## Extracted JSON:\n```json"
)

