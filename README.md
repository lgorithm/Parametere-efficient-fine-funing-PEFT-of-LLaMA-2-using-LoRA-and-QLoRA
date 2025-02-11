# Parameter-Efficient Fine-Tuning(PEFT) of LLaMA-2 using LoRA and QLoRA

## Overview

This project focuses on fine-tuning Meta's LLaMA 2 model using a conversational dataset to enhance its ability to generate human-like responses. Leveraging advanced techniques like Quantized Low-Rank Adaptation (QLoRA), 4-bit quantization, and the Hugging Face `transformers` library, this implementation ensures efficient and effective training on limited hardware resources.

## Features

- Fine-tuning LLaMA 2 for conversational AI tasks
- QLoRA for parameter-efficient training with 4-bit quantization
- Mixed precision and quantization (BitsAndBytes) for optimized performance
- Dataset loading and preprocessing using `datasets` library
- Training using `SFTTrainer` from `trl`

## Usage

### 1. Load the Dataset

Modify the dataset path or use a Hugging Face dataset:

```python
from datasets import load_dataset
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name)
```

### 2. Load and Configure the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
```

### 3. Fine-Tune the Model

```python
from trl import SFTTrainer
trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
trainer.train()
```

### 4. Generate Responses

```python
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = pipeline("Hello, how can I assist you today?")
print(response)
```

## Performance Optimization

- Use mixed precision (`torch.float16`) for reduced memory usage.
- Enable 4-bit quantization with `BitsAndBytesConfig`.
- Implement QLoRA for memory-efficient fine-tuning.

## Results

After fine-tuning, the model demonstrates improved conversational capabilities, generating more coherent and contextually relevant responses.


