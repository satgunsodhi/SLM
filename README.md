# SLM - Small Language Model for Financial Sentiment Analysis

A transformer-based Small Language Model optimized for financial sentiment analysis and actionable trading signal generation. This project combines GPT-inspired architecture with Low-Rank Adaptation (LoRA) and instruction tuning to create an efficient, domain-specific model for real-time financial market analysis.[1]

## Abstract

Financial markets respond strongly to sentiment in news, analyst reports, and social media, yet traditional sentiment analysis often overlooks the nuances of financial language. While large language models like GPT perform well, their high computational requirements limit real-world deployment. This project introduces a lightweight SLM specifically designed for financial sentiment analysis that classifies text as positive, negative, or neutral, then maps these sentiments to actionable trading decisions: buy, sell, or hold.[1]

## Problem Statement

This project addresses five critical gaps in financial sentiment analysis:

**Computational Efficiency**: Large language models are unsuitable for real-time financial applications requiring quick decision-making[1]

**Domain Specificity**: Generic NLP models fail to capture financial jargon, implicit cues, and context-specific sentiment shifts[1]

**Linguistic Complexity**: Traditional lightweight methods cannot handle negations, sarcasm, or domain-specific terminology like "bearish" and "bullish"[1]

**Data Scarcity**: Limited availability of high-quality financial datasets makes full fine-tuning inefficient and prone to overfitting[1]

**Actionable Output**: Most sentiment models stop at polarity classification without translating results into trading signals[1]

## Architecture

### System Overview

The model follows a complete transformer pipeline optimized for financial text processing:[1]

```
Input Layer → Embedding Layer → Transformer Block → Normalization + Output → Prediction
```

### Core Components

**Input Layer**: Tokenized text sequences using GPT-2 tokenizer (Tiktoken) with vocabulary size of 50,257 tokens[1]

**Embedding Layer**: Combines token embeddings with positional encodings to preserve sequence information[1]

**Transformer Block**: Multi-head self-attention mechanism with feed-forward neural networks, dropout regularization, and residual connections[1]

**Normalization**: Layer normalization applied after each sub-layer to stabilize training[1]

**Prediction Layer**: Output logits for next-token prediction and sentiment classification[1]

## Methodology

### Data Pipeline

**Preprocessing & Data Handling**:

- Collection of financial news articles, earnings reports, and market commentary[1]
- Text cleaning removing stopwords, special characters, and numbers[1]
- Domain-specific preprocessing handling financial abbreviations, stock tickers, and numerical expressions[1]
- Tokenization using GPT-2 tokenizer (Tiktoken)[1]

**Dataset Management**:

- Training and validation sets stored in binary NumPy memory-mapped files (train.bin, validation.bin) for efficient I/O[1]
- Token IDs generated via sliding window approach for input-output batch creation[1]
- Randomized batching with PyTorch for robust training[1]
- Primary datasets: FinancialPhraseBank and FiQA via Hugging Face datasets library[1]

### Training Strategy

**Parameter-Efficient Fine-Tuning**: Low-Rank Adaptation (LoRA) reduces computational costs by training only additional low-rank matrices while keeping original model weights frozen[1]

**Instruction Tuning**: Aligns model outputs with human intent, ensuring consistent mapping to standardized sentiment categories (positive, negative, neutral)[1]

**Optimization**: Trained on labeled financial sentiment datasets with careful monitoring to prevent overfitting[1]

### Decision Mapping Layer

The model includes a rule-based + ML mapping layer that converts sentiment classifications into actionable trading signals:[1]

- **Positive** → Buy
- **Negative** → Sell  
- **Neutral** → Hold

Thresholds can be tuned through market backtesting for optimal performance.[1]

## Alternative Approach

The project also explores a **domain adaptation** strategy using pre-trained transformers:[1]

- Leverages FinBERT or LLaMA-2 fine-tuned on financial corpora to capture domain-specific language nuances[1]
- Applies parameter-efficient tuning methods (LoRA, adapters, prompt-tuning) to minimize computational overhead[1]
- Retrieves latest financial news and performs sentiment analysis before mapping to trading decisions[1]

## Technical Implementation

### Requirements

```
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Tiktoken (GPT-2 tokenizer)
- NumPy
- datasets (Hugging Face)
- Jupyter Notebook
```

### Installation

```bash
git clone https://github.com/satgunsodhi/SLM.git
cd SLM
pip install -r requirements.txt
```

### Usage

```python
# Load preprocessed data
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('validation.bin', dtype=np.uint16, mode='r')

# Initialize model with LoRA
model = FinancialSLM(vocab_size=50257, ...)

# Train with instruction tuning
trainer.train(model, train_data, val_data)

# Inference
sentiment = model.predict("Apple reports record earnings beating estimates")
action = decision_mapper(sentiment)  # Returns: "Buy"
```

## Theoretical Foundation

### Transformer Architecture

Built upon Vaswani et al.'s (2017) "Attention is All You Need", the model uses self-attention mechanisms to capture long-range dependencies and subtle contextual cues in financial language. The architecture excels at distinguishing between phrases like "profits fell slightly" versus "profits crashed".[1]

### LoRA (Low-Rank Adaptation)

Following Hu et al. (2021), LoRA enables efficient fine-tuning by training only small additional parameter matrices, significantly reducing computational costs while maintaining model expressiveness. This is particularly valuable for financial datasets which tend to be smaller and domain-specific.[1]

### Instruction Tuning

Based on InstructGPT (Ouyang et al., 2022), instruction tuning aligns model outputs with human intent by training on instruction-formatted datasets. This ensures the model consistently produces standardized sentiment categories suitable for automated trading systems.[1]

### Domain Adaptation

Inspired by Zhang et al. (2023) and Echambadi (2024), the project incorporates retrieval-augmented generation and domain-specific fine-tuning, achieving 15-48% improvements in accuracy and F1 scores compared to generic models.[1]

## Applications

- Real-time financial news sentiment monitoring
- Automated trading signal generation
- Portfolio risk assessment based on sentiment analysis
- Market trend prediction from analyst reports
- Social media sentiment tracking for stocks

## Authors

**Satgun Singh Sodhi** (23BCE0416)  
**Ayushman Mittal** (23BCE0448)[1]

GitHub: [@satgunsodhi](https://github.com/satgunsodhi)

## References

- Vaswani et al. (2017). Attention is All You Need[1]
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models[1]
- Ouyang et al. (2022). Training language models to follow instructions with human feedback[1]
- Zhang et al. (2023). Enhancing Financial Sentiment Analysis via Retrieval-Augmented Large Language Models[1]
- Echambadi (2024). Financial Market Sentiment Analysis Using Large Language Models and Retrieval-Augmented Generation[1]

## License

This project is open source and available for educational and research purposes.
