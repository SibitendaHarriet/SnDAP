
# Sentiment Analysis: A Comprehensive Workflow

This document outlines a multi-model approach to sentiment analysis using various tools and libraries, including TextBlob, VADER, BERT, SentiWordNet, and Llama 3. The goal is to classify text data into sentiments such as **positive**, **neutral**, or **negative**. 


## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Sentiment Analysis Techniques](#sentiment-analysis-techniques)
    - [TextBlob](#textblob)
    - [VADER](#vader)
    - [BERT](#bert)
    - [SentiWordNet](#sentiwordnet)
    - [Llama 3](#llama-3)
4. [Comparison of Results](#comparison-of-results)
5. [Fine tunning on Common Sentiment Agreement](#common-sentiment-agreement)
6. [Visualization and Analysis](#visualization-and-analysis)
7. [Conclusion](#conclusion)

---

## 1. Introduction <a name="introduction"></a>
Sentiment analysis identifies the sentiment of text as **positive**, **negative**, or **neutral**. This document uses both lexicon-based and machine-learning-based approaches.  

**Dataset**: We process a dataset containing text in the `Text_translate2` column.  

---

## 2. Data Preprocessing <a name="data-preprocessing"></a>
Before applying sentiment analysis, the data is preprocessed to remove noise, tokenize text, and handle missing values.  

### Key Preprocessing Steps:
```python
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv('/home/hsibitenda/scratch/harriet/llama2_and_themes.csv')

# Convert all columns to string
df = df.astype(str)

# Tokenizing the text
df['tokenized_text'] = df['Text_translate2'].apply(lambda x: x.split())

# Vocabulary Counter
from collections import Counter
vocab = Counter()
for tokens in df['tokenized_text']:
    vocab.update(tokens)
print("Most common words:", vocab.most_common(10))
```

### Output Example:
The `vocab` counter provides the most common words in the dataset, which helps analyze the text distribution.

---

## 3. Sentiment Analysis Techniques <a name="sentiment-analysis-techniques"></a>

### 3.1 TextBlob <a name="textblob"></a>
TextBlob is a lexicon-based approach to calculate **polarity** and **subjectivity** scores.

#### Code Snippet:
```python
from textblob import TextBlob

# Add sentiment columns
for row in df.itertuples():
    text = row.Text_translate2
    analysis = TextBlob(text)
    df.at[row.Index, 'polarity'] = analysis.sentiment.polarity
    df.at[row.Index, 'subjectivity'] = analysis.sentiment.subjectivity

    if analysis.sentiment.polarity > 0:
        df.at[row.Index, 'Sentiment'] = "Positive"
    elif analysis.sentiment.polarity < 0:
        df.at[row.Index, 'Sentiment'] = "Negative"
    else:
        df.at[row.Index, 'Sentiment'] = "Neutral"
```


### 3.2 VADER <a name="vader"></a>
VADER is tailored for social media text. It calculates positive, negative, and neutral scores.

#### Code Snippet:
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()
for index, text in df['Text_translate2'].items():
    score = analyzer.polarity_scores(text)
    df.loc[index, 'neg'] = score['neg']
    df.loc[index, 'neu'] = score['neu']
    df.loc[index, 'pos'] = score['pos']
    df.loc[index, 'compound'] = score['compound']
    df.loc[index, 'Sentiment_vad'] = (
        "Positive" if score['pos'] > score['neg'] else 
        "Negative" if score['neg'] > score['pos'] else "Neutral"
    )
```

---

### 3.3 BERT <a name="bert"></a>
A transformer-based model (`bert-base-multilingual-uncased-sentiment`) predicts sentiment from 1 to 5 stars.

#### Code Snippet:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment').to('cuda')

def sentiment_score(text):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True).to('cuda')
    outputs = model(**inputs)
    return torch.argmax(outputs.logits) + 1

df['Sentiment_bert'] = df['Text_translate2'].apply(sentiment_score)
```

---

### 3.4 SentiWordNet <a name="sentiwordnet"></a>
Uses lexicons for word-level sentiment analysis.

#### Code Snippet:
```python
from nltk.corpus import sentiwordnet as swn
nltk.download('sentiwordnet')
import spacy

nlp = spacy.load("en_core_web_sm")

def get_swn_score(text):
    doc = nlp(text)
    total_score = sum(
        synset.pos_score() - synset.neg_score()
        for token in doc
        for synset in swn.senti_synsets(token.text)
    )
    return "Positive" if total_score > 0 else "Negative" if total_score < 0 else "Neutral"

df['Sentiment_swn'] = df['Text_translate2'].apply(get_swn_score)
```

---

### 3.5 Llama 3 <a name="llama-3"></a>
A generative language model generates sentiment using predefined prompts.

#### Code Snippet:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct').to('cuda')

prompt = "Analyze sentiment as positive, negative, or neutral.\nText: \"{}\"\nSentiment:"
def evaluate(text):
    inputs = tokenizer(prompt.format(text), return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

df['llama3_sentiment'] = df['Text_translate2'].apply(evaluate)
```

---

## 4. Comparison of Results <a name="comparison-of-results"></a>
After applying all models, compare the results for consistency.

```python
# Add comparison column
df['Common_Sentiment'] = df[['Sentiment', 'Sentiment_vad', 'Sentiment_bert', 'Sentiment_swn', 'llama3_sentiment']].mode(axis=1)[0]
```

---
To address the missing sections about fine-tuning BERT for the task, let's break it down into key steps:

---

## 5. Fine tunning on Common Sentiment Agreement <a name="#common-sentiment-agreement"></a>
###  **Fine-tuning Overview**
Fine-tuning involves training the pre-trained BERT model on your specific dataset (in this case, text sentiment classification). During this process, weights are updated to learn task-specific features.

###  **Preparing Data for Fine-Tuning**
Ensure that the input text data is properly tokenized and converted into tensors compatible with the BERT model. This is already implemented in the provided code. The key steps include:

1. **Text Tokenization:** Using `BertTokenizer` to convert text into token IDs.
2. **Padding/Truncation:** Padding token sequences to a fixed maximum length (`MAX_LEN`).
3. **Attention Masks:** Creating masks to differentiate padded tokens from real tokens.

---

###  **Fine-Tuning with BERT**
The main missing steps are the **loss calculation** and **model evaluation** during the training process. Below is a step-by-step implementation:

---

#### **3.1 Setting Up the BERT Model**
```python
from transformers import BertForSequenceClassification, AdamW

# Load BERT pre-trained model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Pre-trained BERT model
    num_labels=len(np.unique(labels_balanced))  # Number of output classes
)

# Move the model to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

#### **3.2 Defining the Optimizer**
The optimizer is configured for fine-tuning:
```python
# Use AdamW optimizer for better weight regularization
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
```

---

#### **3.3 Training Loop for Fine-Tuning**
The training loop updates model weights and evaluates the validation accuracy:

```python
from tqdm import trange

# Set number of epochs and track loss
epochs = 4
train_loss_set = []

# Function to calculate accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training loop
for epoch in trange(epochs, desc="Epoch"):
    # Training phase
    model.train()
    tr_loss = 0  # Track training loss

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)  # Move tensors to GPU/CPU
        b_input_ids, b_input_mask, b_labels = batch

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Update loss
        tr_loss += loss.item()

    # Print training loss
    print(f"Epoch {epoch + 1} - Train loss: {tr_loss / len(train_dataloader)}")

    # Validation phase
    model.eval()
    eval_accuracy = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)  # Move tensors to GPU/CPU
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits

        # Calculate validation accuracy
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print(f"Validation Accuracy: {eval_accuracy / nb_eval_steps}")
```

---

###  **Evaluation**
Once fine-tuning is complete, evaluate the model's performance on the validation or test set.

```python
from sklearn.metrics import classification_report

# Evaluate the model
model.eval()
predictions, true_labels = [], []

for batch in validation_dataloader:
    batch = tuple(t.to(device) for t in batch)  # Move tensors to GPU/CPU
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs.logits

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to("cpu").numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

# Flatten predictions and labels
flat_predictions = [item for sublist in predictions for item in sublist]
flat_true_labels = [item for sublist in true_labels for item in sublist]

# Convert predictions to label ids
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Classification report
report = classification_report(flat_true_labels, flat_predictions)
print(report)
```

---

###  **Training Loss Visualization**
To visualize the training process:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()
```

---

###  **Saving and Reloading the Fine-Tuned Model**
After fine-tuning, you can save the model for later use:
```python
# Save the model
model.save_pretrained("bert_finetuned_model")
tokenizer.save_pretrained("bert_finetuned_model")

# Reload the model
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("bert_finetuned_model")
tokenizer = BertTokenizer.from_pretrained("bert_finetuned_model")
```

## Fine tunning on LLMs
###  Hugging Face Token Setup

The first part of the code is dedicated to securely logging into Hugging Face. Here's how it's structured:

```python
import getpass
import os
import subprocess
from huggingface_hub import notebook_login

# Define path to Hugging Face token file
token_file_path = "/home/hsibitenda/scratch/harriet/llms/huggingface_token.txt"

if os.path.exists(token_file_path):
    with open(token_file_path, "r") as file:
        token = file.read().strip()
else:
    print("Please enter your Hugging Face token:")
    token = getpass.getpass()

# Set token environment variable for Hugging Face
os.environ["HUGGINGFACE_TOKEN"] = token

# Log into Hugging Face
command = f'transformers-cli login --token {token}'
subprocess.run(command, shell=True)
```

**Explanation:**
- **Token Security**: The token is stored in a secure file (`huggingface_token.txt`). If the file doesn’t exist, it prompts the user for input using `getpass.getpass()` to prevent exposure.
- **Login Process**: `transformers-cli login` is used to authenticate against Hugging Face. This allows for easy access to hosted models and datasets.

---

### **. Model Loading and Preprocessing**

In this section, the model is loaded, and a tokenizer is initialized. The script uses **LLaMA 3**, which is a pre-trained language model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define model ID
model_id = "meta-llama/Meta-Llama-3-8B"

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=128)
tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token

# Load the LLaMA 3 model
model = AutoModelForCausalLM.from_pretrained(model_id)
model.to(device)
model.eval()
```

**Explanation:**
- **Tokenizer Setup**: The tokenizer prepares the text to be fed into the model. It ensures that the tokenized input does not exceed a specified length (`max_length=128`).
- **Model Setup**: We load the pre-trained LLaMA 3 model for causal language modeling (`AutoModelForCausalLM`). The model is then moved to the appropriate device (GPU or CPU).

---

### ** Fine-Tuning with LoRA**

**Low-Rank Adaptation (LoRA)** is a technique for efficient fine-tuning of large models. Instead of training all parameters, LoRA introduces trainable low-rank matrices to adapt the model while keeping the pre-trained weights fixed.

```python
from peft import LoraConfig
from transformers import TrainingArguments

# LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16, 
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules="lm_head",  # Adapt the lm_head
    task_type="CAUSAL_LM",
)

# Training Arguments
training_arguments = TrainingArguments(
    output_dir="./trained_weights",  # Where to save model
    num_train_epochs=3,  # Number of epochs
    per_device_train_batch_size=1,  # Batch size per device
    gradient_accumulation_steps=8,  # Accumulate gradients for memory efficiency
    gradient_checkpointing=True,  # Enable gradient checkpointing
    optim="paged_adamw_32bit",  # Memory-efficient optimizer
    logging_steps=25,  # Log every 25 steps
    learning_rate=2e-4,  # Learning rate
    fp16=True,  # Use FP16 for faster training
    save_steps=0,  # Save model after each epoch
    evaluation_strategy="epoch"  # Evaluate after every epoch
)
```

**Explanation:**
- **LoRA Setup**: This configuration ensures that only the low-rank adaptation matrices are updated during fine-tuning. The `target_modules="lm_head"` means that we adapt the final output layer.
- **Training Arguments**: These define how the model will be trained, including batch size, gradient accumulation, memory-saving strategies (e.g., `gradient_checkpointing`, `fp16`), and the optimizer (`paged_adamw_32bit`).

---

### ** Training the Model**

We use the `SFTTrainer` class from the `trl` library for supervised fine-tuning:

```python
from trl import SFTTrainer
from datasets import Dataset

train_data = Dataset.from_pandas(X_train)  # Convert pandas DataFrame to Hugging Face Dataset
eval_data = Dataset.from_pandas(X_eval)

# Initialize trainer with model, arguments, datasets, and LoRA configuration
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    tokenizer=tokenizer,
    max_seq_length=128,
)
trainer.train()  # Start training
```

**Explanation:**
- **Data Conversion**: We convert our `pandas` DataFrames (`X_train` and `X_eval`) into Hugging Face `Dataset` format for efficient handling during training.
- **Training Process**: The `SFTTrainer` is a specialized trainer class designed to handle supervised fine-tuning tasks. The trainer will handle loss computation, backward pass, and model updates.

---

### ** Prediction and Evaluation**

After fine-tuning, we use the model for predictions on the test set:

```python
def predict(test, model, tokenizer, generator, device):
    y_preds = []
    for i in range(len(test)):
        prompt = test.iloc[i]["Text_translate2"]  # Prepare prompt text
        prompt = prompt[:128]  # Truncate to max length
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # Tokenize input

        result = generator(input_ids=input_ids, max_length=128)
        prediction = result[0]["generated_text"].split("=")[-1].strip()

        # Map predictions to sentiment labels
        if "positive" in prediction:
            y_preds.append("positive")
        elif "neutral" in prediction:
            y_preds.append("neutral")
        elif "negative" in prediction:
            y_preds.append("negative")
        else:
            y_preds.append("neutral")  # Default if prediction is unclear
    return y_preds
```

**Explanation:**
- **Input Preparation**: The input text (`Text_translate2`) is tokenized into `input_ids`, ensuring the input fits the model's max token length.
- **Prediction**: The model generates predictions for the given prompt. These predictions are then parsed, and sentiment labels are assigned based on keywords like `positive`, `neutral`, or `negative`.

---

### ** Evaluation and Metrics Calculation**

After predictions are made, we can evaluate the model’s performance using various metrics:

```python
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
```

**Explanation:**
- **Metrics**: The `accuracy_score` gives a general idea of how well the model performs, while the `classification_report` provides a detailed analysis of precision, recall, and F1-score for each class.
- **Confusion Matrix**: This shows the performance of the classifier, illustrating the number of true positives, false positives, true negatives, and false negatives for each class.

---

### ** Saving the Fine-Tuned Model**

Once the model is fine-tuned, it’s important to save the model and tokenizer:

```python
# Save model and tokenizer after fine-tuning
trainer.save_model()
tokenizer.save_pretrained("./finetuned_model")
```

**Explanation:**
- **Saving**: The model and tokenizer are saved to the specified directory (`./finetuned_model`) so that they can be loaded later for inference or deployment.

---

### **8. Post-Training Cleanup**

After fine-tuning, it's often necessary to clear memory to prevent any unnecessary overhead:

```python
import gc
torch.cuda.empty_cache()
gc.collect()

from peft import AutoPeftModelForCausalLM

finetuned_model = "/home/hsibitenda/scratch/harriet/sentiments/trained_weights2"
compute_dtype = getattr(torch, "float16")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

model = AutoPeftModelForCausalLM.from_pretrained(
     finetuned_model,
     torch_dtype=compute_dtype,
     return_dict=False,
     low_cpu_mem_usage=True,
     device_map=device,
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model2",safe_serialization=True, max_shard_size="2GB")
tokenizer.save_pretrained("./merged_model2")
```

**Explanation:**
- **Memory Cleanup**: This ensures that all GPU memory is freed up and helps prevent memory leaks, especially in long-running sessions.

---

### **9. Batch Processing for Large Datasets**

The following snippet demonstrates how to split a large dataset into smaller batches to manage memory and ensure efficient processing:
#### Generating Predictions
Now that the data is preprocessed and the model is ready, it’s time to run predictions on the new, unseen dataset. We'll use the model’s generate method to produce the sentiment prediction for each input text. This step also involves parsing the model’s output.

```python
def predict_sentiment(texts, model, tokenizer, device):
    # Prepare list to store predictions
    predictions = []
    
    # Process each text input for sentiment prediction
    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        
        # Generate the model's prediction for the given text
        output = model.generate(input_ids, max_length=128, num_return_sequences=1)

        # Decode the generated text and extract sentiment
        predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Based on the generated text, map it to a sentiment class
        if "positive" in predicted_text.lower():
            predictions.append("positive")
        elif "neutral" in predicted_text.lower():
            predictions.append("neutral")
        elif "negative" in predicted_text.lower():
            predictions.append("negative")
        else:
            predictions.append("neutral")  # Default class if no sentiment found
    
    return predictions

# Call the function on your new data
predicted_sentiments = predict_sentiment(texts_to_predict, model, tokenizer, device)

# Adding predictions to the dataframe

df_new_data['Predicted Sentiment'] = predicted_sentiments
```
Explanation:

Prediction Process:
We tokenize each input text and pass it through the model.
The generate method produces a sequence of tokens representing the model's prediction for the input. We decode this output back into text.
We look for keywords like "positive", "neutral", or "negative" in the generated text and assign the corresponding sentiment label. If the model is unclear, we default to "neutral".
Data Integration: The predictions are stored in a list and then added as a new column (Predicted Sentiment) to the original DataFrame.

```python
# Split dataframe into two parts
df1 = df.sample(frac=0.5, random_state=42)
df2 = df.drop(df1.index)

# Save the resulting dataframes
df1.to_csv("batch1.csv")
df2.to_csv("batch2.csv")
```

## 5. Visualization <a name="visualization-and-analysis"></a>
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot sentiment distribution
sns.countplot(data=df, x='Common_Sentiment')
plt.title('Sentiment Distribution Across Models')
plt.show()
```
**Explanation:**
- **Data Splitting**: The dataset (`df`) is split into two parts using `sample(frac=0.5)`. This ensures that we can process large datasets in manageable chunks, reducing memory overhead during prediction.

---

### Conclusion

This pipeline enables efficient fine-tuning, evaluation, and deployment of a large language model for sentiment analysis. By using techniques like **LoRA** and **gradient checkpointing**, the code ensures that fine-tuning large models can be done with less computational cost. 


