# Generating Joint Entities and Relation Types

This document provides a comprehensive guide on extracting entities and their relationships from text data using the **REBEL model**, integrating the output into a structured knowledge base, and further processing and categorizing the results using Python and **spaCy**.

### Key Concepts
1. **Entities**: Named objects in text (e.g., people, places, organizations).
2. **Relations**: Connections or relationships between entities (e.g., "is located in," "is part of").

### Workflow
1. Extract relations using the REBEL model.
2. Build a knowledge base (KB) to store entities and relationships.
3. Categorize entities using **spaCy** Named Entity Recognition (NER).
4. Process and store results in structured formats for analysis.

---

## Code Snippets and Explanations

### 1. **Initial Setup**

Set up the required libraries and load the models for text processing.

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
import pandas as pd
from ast import literal_eval

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Load the REBEL model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to(device)
```

---

### 2. **Relation Extraction**

Define a function to decode the REBEL model's output into structured relationships.

```python
def extract_relations_from_model_output(text):
    relations = []
    relation, subject, object_ = '', '', ''
    current = None
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")

    for token in text_replaced.split():
        if token == "<triplet>":
            if relation:
                relations.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation, subject, object_ = '', '', ''
            current = 'subject'
        elif token == "<subj>":
            current = 'subject'
        elif token == "<obj>":
            current = 'object'
        else:
            if current == 'subject':
                subject += f" {token}"
            elif current == 'object':
                object_ += f" {token}"
            else:
                relation += f" {token}"

    if subject and relation and object_:
        relations.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
    
    return relations
```

---

### 3. **Processing Text**

Process text data using the REBEL model, extract relations, and structure them into a knowledge base.

```python
class KB:
    def __init__(self):
        self.entities = {}
        self.relations = []

    def add_entity(self, entity_name):
        if entity_name not in self.entities:
            self.entities[entity_name] = {"title": entity_name}

    def add_relation(self, relation):
        if relation not in self.relations:
            self.relations.append(relation)

def from_text_to_kb(text, span_length=128):
    inputs = tokenizer([text], return_tensors="pt", max_length=span_length, truncation=True).to(device)
    generated_tokens = model.generate(**inputs, max_length=64, num_beams=3, num_return_sequences=3)
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    kb = KB()
    for pred in decoded_preds:
        relations = extract_relations_from_model_output(pred)
        for relation in relations:
            kb.add_entity(relation["head"])
            kb.add_entity(relation["tail"])
            kb.add_relation(relation)
    
    return kb
```

---

### 4. **Entity Categorization with spaCy**

Categorize entities using spaCy's NER for additional metadata.

```python
def extract_entity_categories(text):
    doc = nlp(str(text))
    entity_categories = set()

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            entity_categories.add('persons')
        elif ent.label_ == 'ORG':
            entity_categories.add('organisations')
        elif ent.label_ == 'GPE':
            entity_categories.add('locations')

    return list(entity_categories)
```
---

### 5. **Processing and Expanding Relations**

Process the extracted relations, expand them into a detailed format, and save them to a new DataFrame.
![Processing and Expanding Relations](../entites_0.png)
```python
# Load the original DataFrame
df_original = pd.read_csv('path_to_original_file.csv')

# List to store expanded rows
rows = []

# Iterate through each row in the original DataFrame
for index, row in df_original.iterrows():
    relations = literal_eval(row['Relations'])

    for relation_entry in relations:
        rows.append({
            'sourcetype': row['sourcetype'],
            'Views': row['Views'],
            'Date': row['Date'],
            'textid': row['textid'],
            'alltext': row['alltext'],
            'llama2_labelone': row['llama2_labelone'],
            'cleaned_theme': row['cleaned_theme'],
            'llama2_labelone10': row['llama2_labelone10'],
            'Predicted_Categories3': row['Predicted_Categories3'],
            'llama3_sentiment': row['llama3_sentiment'],
            'Text_lemma': row['Text_lemma'],
            'Entities': row['Entities'],
            'Relations': row['Relations'],
            'Entity_Categories': row['Entity_Categories'],
            'Head': relation_entry['head'],
            'Relation_Type': relation_entry['type'],
            'Tail': relation_entry['tail'],
            'Spans': relation_entry['meta']['spans']
        })

new_df['Entity_Categories2'] = new_df.apply(
    lambda row: extract_entity_categories((row['Head'], row['Relation_Type'], row['Tail'])), axis=1
)
```
![Processing and Expanding Relations](../entities_02.png)

### Evaluating Categories of Entities Generated

This section evaluates the performance of a BERT-based classification model to predict entity categories. The process is broken into manageable steps:

---

#### Step 1: Dataset Preparation

We start by preparing the dataset, mapping entity categories to numerical labels, and splitting it into training and testing sets.

```python
from sklearn.model_selection import train_test_split

# Convert the 'Entity_Categories2' column to strings and map them to numerical labels
df['category'] = df['Entity_Categories2'].apply(lambda x: str(x))
df['label_id'] = df['category'].astype('category').cat.codes

# Split dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")
```

---

#### Step 2: BERT Tokenization and Dataset Class

We create a custom dataset class for tokenizing input data and preparing it for the BERT model.

```python
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class CustomDataset(Dataset):
    def __init__(self, df, max_length=128):
        self.df = df
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Tokenize the input text
        inputs = tokenizer(
            self.df.iloc[idx]["Text_lemma"],  # Text input column
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Retrieve the label
        label = torch.tensor(self.df.iloc[idx]["label_id"]).long()
        return {"input_ids": inputs["input_ids"].squeeze(), 
                "attention_mask": inputs["attention_mask"].squeeze(), 
                "label": label}
```

---

#### Step 3: BERT Model Initialization

Initialize a BERT model for sequence classification, configured with the number of unique labels.

```python
from transformers import BertForSequenceClassification

# Count the unique labels for classification
num_labels = len(df['label_id'].unique())

# Load the BERT model for classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

print(f"BERT model initialized with {num_labels} labels.")
```

---

#### Step 4: Training in Chunks

The training process is conducted in chunks to handle memory constraints effectively.

```python
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_joint_entity_relation",
    per_device_train_batch_size=8,
    save_total_limit=1,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=100,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
)

# Train the model in chunks
chunk_size = 1000
for start in range(0, len(train_df), chunk_size):
    end = start + chunk_size
    chunk_train_df = train_df.iloc[start:end]
    chunk_train_dataset = CustomDataset(chunk_train_df)
    trainer.train_dataset = chunk_train_dataset
    trainer.train()

print("Model training completed.")
```

---

#### Step 5: Evaluation and Metrics

Evaluate the trained model on the test dataset and calculate performance metrics.

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Prepare the test dataset
test_dataset = CustomDataset(test_df)

# Evaluate the model
results = trainer.evaluate(test_dataset)

# Generate predictions
predictions = trainer.predict(test_dataset)
predicted_labels = torch.argmax(torch.from_numpy(predictions.predictions), dim=1).tolist()
true_labels = test_df['label_id'].tolist()

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
```
![Step 5: Evaluation and Metrics](../entities_01.png)
---

### Viewing Entities for Specific Topics

This demonstrates how to analyze and visualize top entities for specific topics.

```python
df['location_entity'] = df['Head'] + ', ' + df['Tail']
filtered_df = df[df['Entity_Categories2'].apply(lambda x: 'locations' in x)]
grouped = filtered_df.groupby(['llama2_labelone', 'location_entity']).size().reset_index(name='count')
```

#### Top Topics & locations

```python
topic_counts = grouped.groupby('llama2_labelone').size()
filtered_topics = topic_counts[(topic_counts >= 20) & (topic_counts <= 23)].index
filtered_data = grouped[grouped['llama2_labelone'].isin(filtered_topics)]

# Plotting
```
![Top Topics & Plot](../enties1.png)

#### Top topics and Persons Entities

1. **Filter & Group**: Filter for `persons` and group similarly.

```python
df['persons_entity'] = df['Head'] + ', ' + df['Tail']
filtered_df = df[df['Entity_Categories2'].apply(lambda x: 'persons' in x)]
grouped = filtered_df.groupby(['llama2_labelone', 'persons_entity']).size().reset_index(name='count')
```

2. **Top Topics & Plot**: Visualize top 5 person entities.

```python
filtered_topics = grouped.groupby('llama2_labelone').size().between(50, 150).index
filtered_data = grouped[grouped['llama2_labelone'].isin(filtered_topics)]
top_topics = filtered_data['llama2_labelone'].value_counts().head(10).index

# Plot
```
![Top topics and Persons Entities](../enties2.png)


#### Top topics and Organisations Entities

1. **Filter & Group**: Filter for `organisations` and group similarly.

```python
df['organisations_entity'] = df['Head'] + ', ' + df['Tail']
filtered_df = df[df['Entity_Categories2'].apply(lambda x: 'organisations' in x)]
grouped = filtered_df.groupby(['llama2_labelone', 'organisations_entity']).size().reset_index(name='count')
```

2. **Top Topics & Plot**: Visualize the top 5 organization entities.

```python
filtered_topics = grouped.groupby('llama2_labelone').size().between(50, 210).index
filtered_data = grouped[grouped['llama2_labelone'].isin(filtered_topics)]
top_topics = filtered_data['llama2_labelone'].value_counts().head(10).index

# Plot
```
![Top topics and Organisations Entities](../entities3.png)
