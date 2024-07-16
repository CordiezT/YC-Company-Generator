# Fine-Tuning GPT-2 for Business Descriptions

This script fine-tunes a GPT-2 model to generate startup business descriptions using provided training data. Ideal for entrepreneurs, data scientists, and AI enthusiasts looking to create unique and tailored business content.

## Requirements

- Python 3.x
- PyTorch
- Transformers

## Installation

1. **Install dependencies**:
    ```sh
    pip install torch transformers
    ```

2. **Prepare your data**:
    - Place your training data file (`yc_startup_descriptions.txt`) in the script directory.

## Script Overview

### 1. Environment Setup
The script begins by checking if CUDA is available to utilize GPU for faster training:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 2. Model and Tokenizer Loading
Load the GPT-2 model and tokenizer from the Hugging Face library:
```python
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.to(device)
```

### 3. Dataset Loading
Define a function to load the dataset:
```python
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )
```
Load the training data:
```python
train_dataset = load_dataset('yc_startup_descriptions.txt', tokenizer)
```

### 4. Data Collation
Set up the data collator for language modeling:
```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
```

### 5. Training Setup
Specify training arguments such as output directory, number of epochs, batch size, and saving steps:
```python
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)
```

### 6. Trainer Initialization
Initialize the `Trainer` with the model, arguments, data collator, and dataset:
```python
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)
```

### 7. Training the Model
Train the model and save the fine-tuned version:
```python
trainer.train()
model.save_pretrained('./fine_tuned_gpt2')
tokenizer.save_pretrained('./fine_tuned_gpt2')
```

### 8. Text Generation
Load the fine-tuned model and set up the text generation pipeline:
```python
fine_tuned_model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_gpt2')
generator = pipeline('text-generation', model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
```
Generate business descriptions based on given prompts:
```python
prompts = ["Our startup is focused on", "We are developing", "Our company specializes in"]
descriptions = generator(prompts, max_length=50, num_return_sequences=10)
```
Print the generated descriptions:
```python
for i, desc in enumerate(descriptions):
    print(f"Description {i+1}: {desc['generated_text']}\n")
```

## Usage

1. **Run the script** to train the model and generate business descriptions:
    ```sh
    python your_script.py
    ```

2. **View Output**: The script will output generated business descriptions based on the provided prompts.

## Customization

- **Training Data**: Replace `yc_startup_descriptions.txt` with your own training data file.
- **Prompts**: Modify the `prompts` list to include different starting phrases for business descriptions.
- **Generation Parameters**: Adjust `max_length` and `num_return_sequences` in the `generator` call to change the length and number of generated descriptions.

## Example Output

```
Description 1: Our startup is focused on leveraging AI to optimize supply chain logistics...
Description 2: We are developing a new platform for remote team collaboration...
...
```

## Conclusion

This script provides a powerful tool for fine-tuning a GPT-2 model to generate high-quality, customized business descriptions. Perfect for startups looking to articulate their vision or for anyone interested in exploring the capabilities of AI in natural language generation.
