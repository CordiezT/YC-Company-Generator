import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Check if CUDA is available and use GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.to(device)

# Function to load the dataset
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

# Load your descriptions from the text file
train_dataset = load_dataset('yc_startup_descriptions.txt', tokenizer)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_gpt2')
tokenizer.save_pretrained('./fine_tuned_gpt2')

# Load the fine-tuned model for generation
fine_tuned_model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_gpt2')

# Set up the text generation pipeline
from transformers import pipeline
generator = pipeline('text-generation', model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

# Generate business descriptions
prompts = ["Our startup is focused on", "We are developing", "Our company specializes in"]
descriptions = generator(prompts, max_length=50, num_return_sequences=10)

for i, desc in enumerate(descriptions):
    print(f"Description {i+1}: {desc['generated_text']}\n")
