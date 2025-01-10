I see! Let's carefully go through the sections to ensure nothing is missing and that everything aligns properly with your project's structure. Here's a more complete version of the `README.md`, making sure all components are included:

---

```markdown
# Text Classification and Topic Modeling with BERT, TF-IDF, and Llama-2

This project demonstrates the steps for extracting topics from a corpus, evaluating the quality of topics, and using these topics for classification using machine learning metrics. It involves text preprocessing, topic extraction, evaluation using coherence and divergence, and model training for classification using various machine learning algorithms.

## Prerequisites

Ensure you have the following Python libraries installed:

```bash
pip install numpy pandas tensorflow scikit-learn nltk matplotlib seaborn sentence-transformers hdbscan transformers umap-learn
```

## 1. **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
```

## 2. **Loading and Preprocessing Data**

Read the dataset, preprocess the text by removing stopwords, and prepare the text data for modeling.

```python
# Load the dataset
df = pd.read_csv('../harriet/traindataset.csv')

# Preprocess the data
df = df.astype(str)

# Define custom stopwords
custom_stopwords = set(stopwords.words('english'))

# Remove custom stopwords
def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word not in custom_stopwords)

df["Text_lemma"] = df["Text_lemma"].apply(remove_stopwords)
```

## 3. **Tokenization and Frequency Analysis**

Tokenize the documents and perform frequency analysis to identify common words.

```python
# Tokenize the documents
tokenized = df['Text_lemma'].apply(lambda x: x.split())
tokenized_docs = tokenized.values

# Perform frequency analysis
from collections import Counter
cnt = Counter()
for text in df["Text_lemma"].values:
    for word in text.split():
        cnt[word] += 1
most_common_words = [word for word, _ in cnt.most_common(20)]
custom_stopwords.update(most_common_words)
```

## 4. **BERT Embeddings for Text Representation**

Generate embeddings using the `Sentence-Transformers` library for each document.

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Load the model and tokenizer
MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
embedding_model = SentenceTransformer(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize the documents
encoding = tokenizer(
    text=df['Text_lemma'].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    max_length=200,
    truncation=True,
    pad_to_max_length=True,
    verbose=True
)

# Generate embeddings
embeddings = embedding_model.encode(df['Text_lemma'].tolist(), convert_to_tensor=True)
embeddings_cpu = embeddings.cpu().numpy()
```

## 5. **Clustering Using HDBSCAN**

Perform clustering on the embeddings using HDBSCAN.

```python
import hdbscan
import umap

# Apply UMAP for dimensionality reduction
X_embedded = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine').fit_transform(embeddings_cpu)

# Perform HDBSCAN clustering
cluster = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom').fit(X_embedded)
```

## 6. **Topic Labeling with Llama-2 Model**

Use Llama-2 to generate topic labels by logging into Hugging Face and interacting with the model.

```python
import os
import subprocess
import getpass
from huggingface_hub import notebook_login

# Authenticate to Hugging Face
token_file_path = "/path/to/huggingface_token.txt"
if os.path.exists(token_file_path):
    with open(token_file_path, "r") as file:
        token = file.read().strip()
else:
    token = getpass.getpass("Please enter your Hugging Face token:")

os.environ["HUGGINGFACE_TOKEN"] = token
command = f'transformers-cli login --token {token}'
subprocess.run(command, shell=True)

# Set up the Llama-2 model
from torch import cuda
import transformers

model_id = 'meta-llama/Llama-2-7b-chat-hf'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# Generate topic labels using the pre-defined prompts
generator = transformers.pipeline(
    model=model, tokenizer=tokenizer, task='text-generation', temperature=0.1, max_new_tokens=500, repetition_penalty=1.1
)
```

## 7. **Generating Topic Labels**

Generate topic labels by combining the top words for each topic with predefined prompts.

```python
# Generate labels
def generate_labels(prompt, top_n_words):
    labels = []
    for topic, words in top_n_words.items():
        word_str = ', '.join([word[0] for word in words])
        prompt_with_words = prompt.replace("[KEYWORDS]", word_str)
        label = generator(prompt_with_words)[0]['generated_text'].split('\n')[-1].strip()
        labels.append(label)
    return labels

labels = generate_labels(main_prompt, top_n_words)
docs_per_topic['llama2_labelswsp'] = labels
```

## 8. **Evaluating Topics Based on Coherence and Divergence**

After extracting topics, evaluate their quality using coherence and divergence metrics.

```python
# Evaluate coherence and divergence (example using a custom function)
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_coherence(topic_words, top_n_words):
    similarities = cosine_similarity(topic_words, top_n_words)
    coherence_score = np.mean(similarities)
    return coherence_score

# Example for divergence evaluation
from scipy.spatial.distance import jensenshannon

def evaluate_divergence(topic_distribution, global_distribution):
    divergence_score = jensenshannon(topic_distribution, global_distribution)
    return divergence_score
```

## 9. **Model Preprocessing for Classification**

Scale the embeddings and perform a train-test split.

```python
from sklearn import preprocessing, model_selection

# Scaling X_embedded
X_embedded = preprocessing.scale(X_embedded)

# Defining target variable
y = df[['llama2_labelone']].values

# Splitting the data
x_train, x_test, y_train, y_test = model_selection.train_test_split(X_embedded, y, test_size=0.2, random_state=42, stratify=y)

# Apply UMAP only on the training data
umap_transformer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
x_train = umap_transformer.fit_transform(x_train)

# Transform the test data using the fitted UMAP transformer
x_test = umap_transformer.transform(x_test)
```

## 10. **Model Definitions**

You define a set of machine learning models to train and evaluate.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier, VotingClassifier

# Define models
def get_models():
    models = dict()
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    models['svm'] = SVC()
    models['RF'] = RandomForestClassifier()
    models['bayes'] = GaussianNB()
    models['MLP'] = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300, activation='relu', solver='adam', random_state=1)
    
    return models
```

## 11. **Model Training**

You train each model separately on the training data.

```python
# Train models
bayes = GaussianNB().fit(x_train, y_train)
lr = LogisticRegression().fit(x_train, y_train)
knn = KNeighborsClassifier().fit(x_train, y_train)
cart = DecisionTreeClassifier().fit(x_train, y_train)
svm = SVC(probability=True).fit(x_train, y_train)
RF = RandomForestClassifier().fit(x_train, y_train)
MLP = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300, activation='relu', solver='adam', random_state=1).fit(x_train, y_train)
```

## 12. **Model Evaluation (Accuracy)**

You calculate accuracy and other evaluation metrics for each model.

```python
models = get_models()
for name, model in models.items():
    print(f'Accuracy of {name}: {model.score(x_test, y_test)}')


```

## 13. **Visualizing Results**

You can visualize the classification results with confusion matrices or classification reports.

```python
# Visualize confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
```
## 14. **Model Evaluation Results**

The following box plot displays the distributions of key evaluation metrics (F1 Score, Recall, Balanced Accuracy, Accuracy, and Precision) for the models used in the study. These metrics were evaluated using bootstrapping with 90% resampling of the test data.

![Model Evaluation Metrics](docs/3_Extraction_of_social_concerns/topic_eval2.png)

### Explanation:
- The plot shows the spread of values for each metric across all bootstrap samples. The boxes represent the interquartile range (IQR), and the lines within the boxes indicate the median.
- Each metric provides insights into the model's overall performance, robustness, and stability across various subsets of the test data.

Sure! Let's break the explanation and the code into smaller, more digestible sections for better readability in the `README.md`. Here's a revised version:

---

## 16. **Topic Merging for Smaller Clusters**

In our analysis, we aimed to reduce the number of topics by merging similar ones. This helps consolidate related topics into a smaller set of general clusters, making the results more interpretable and less redundant.

### Overview of the Merging Process

The merging process involves the following steps:

1. **Initial Topic Creation**: The dataset is initially clustered into a certain number of topics.
2. **Cosine Similarity**: We calculate the cosine similarity between the term frequency-inverse document frequency (TF-IDF) vectors of topics to measure how similar each one is.
3. **Merging Criteria**: Topics with high cosine similarity are candidates for merging. The merging continues until the number of topics reaches a target (e.g., 10 topics).
4. **Final Clusters**: After merging, we are left with fewer, more representative topics.

### Step 1: Extracting Top Words for Each Topic

Before merging, we extract the top words for each topic to better understand the clusters. This is done using the TF-IDF matrix.

```python
def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=10):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.cluster_id2)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words
```

This function extracts the top `n` words for each topic based on their TF-IDF scores.

### Step 2: Extracting Topic Sizes

We calculate the sizes of the topics (how many documents belong to each topic) to help decide which topics should be merged.

```python
def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['cluster_id2'])
                     .Text_lemma
                     .count()
                     .reset_index()
                     .rename(columns={"cluster_id2": "Topic", "Text_lemma": "Size"})
                     .sort_values("Size", ascending=False))
    return topic_sizes
```

This function computes the size of each topic by counting the number of documents assigned to it.

### Step 3: Merging Topics Based on Cosine Similarity

Next, we compute the cosine similarity between the TF-IDF vectors of the topics and merge topics that are very similar to each other. We continue this process until we have fewer than 10 topics.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the number of topics
num_topics = df['cluster_id2'].nunique()

while num_topics > 10:
    # Calculate cosine similarity
    similarities = cosine_similarity(tf_idf.T)
    np.fill_diagonal(similarities, 0)

    # Extract the smallest topics to merge
    topic_sizes = df.groupby(['cluster_id2']).count().sort_values("Text_lemma", ascending=False).reset_index()
    topics_to_merge = topic_sizes.iloc[-10:].cluster_id2.tolist()  # Choose the top 10 topics to merge
    
    for topic_to_merge in topics_to_merge:
        if topic_to_merge + 1 < len(similarities):
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1])
            
            # Merge the topics
            df.loc[df.cluster_id2 == topic_to_merge, "cluster_id2"] = topic_to_merge_into
            old_topics = df.sort_values("cluster_id2").cluster_id2.unique()
            map_topics = {old_topic: index for index, old_topic in enumerate(old_topics)}
            df.cluster_id2 = df.cluster_id2.map(map_topics)
        
    docs_per_topic = df.groupby(['cluster_id2'], as_index=False).agg({'Text_lemma': ' '.join'})
    
    # Recalculate TF-IDF and top words
    m = len(embeddings)
    tf_idf, count = c_tf_idf(docs_per_topic.Text_lemma.values, m)
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=10)

    # Update the number of topics
    num_topics = df['cluster_id2'].nunique()

    print(f"Number of topics: {num_topics}")
```

In this code:
- We calculate the cosine similarity matrix for the TF-IDF vectors of each topic.
- We select the smallest topics (based on size) and merge them with the most similar ones.
- This process continues until the number of topics is reduced to 10 or fewer.

### Step 4: Final Topic Sizes and Top Words

After merging, we calculate the final sizes of the topics and extract the top words for each of the remaining topics.

```python
topic_sizes = extract_topic_sizes(df)
top_n_words
```

This gives us an updated list of topic sizes and the most relevant words for each remaining topic after the merging process.
```Number of topics: 295
Number of topics: 286
Number of topics: 277
Number of topics: 268
Number of topics: 260
Number of topics: 252
Number of topics: 244
Number of topics: 236
Number of topics: 228
Number of topics: 220
Number of topics: 212
Number of topics: 203
Number of topics: 194
Number of topics: 185
Number of topics: 176
Number of topics: 168
Number of topics: 160
Number of topics: 152
Number of topics: 146
Number of topics: 140
Number of topics: 135
Number of topics: 130
Number of topics: 125
Number of topics: 120
Number of topics: 115
Number of topics: 110
Number of topics: 105
Number of topics: 99
Number of topics: 93
Number of topics: 87
Number of topics: 81
Number of topics: 75
Number of topics: 69
Number of topics: 63
Number of topics: 56
Number of topics: 50
Number of topics: 46
Number of topics: 41
Number of topics: 38
Number of topics: 34
Number of topics: 29
Number of topics: 22
Number of topics: 16
Number of topics: 10
{0: [('covid', 0.004264753010553059),
  ('african', 0.004132396355870341),
  ('time', 0.004105565743630238),
  ('video', 0.004008177708356928),
  ('nigeria', 0.003935928529495938),
  ('visa', 0.003928455561424404),
  ('medium', 0.0039184777286674615),
  ('passport', 0.003855947504467974),
  ('state', 0.0038488254771742152),
  ('driver', 0.003823655697035549)],
 1: [('frontline', 0.13126725145605064),
  ('pb', 0.08563254354865356),
  ('documentaryconfusion', 0.07623193885681112),
  ('reactionsm', 0.07484238945339566),
  ('spiritualist', 0.07330123792427921),
  ('traveling', 0.06590773702997366),
  ('solved', 0.06149399396263008),
  ('official', 0.05315088473705128),
  ('infertility', 0.05224802639715868),
  ('spirit', 0.051712925028530234)],
 2: [('pitbull', 0.28214703242386613),
  ('washington', 0.19640066856094782),
  ('schumer', 0.159469559757918),
  ('oncamera', 0.15547237511813597),
  ('pelosi', 0.15287422840258238),
  ('shouting', 0.13958144630617214),
  ('post', 0.13751731412005624),
  ('alicia', 0.12267262279298526),
  ('habia', 0.11941674493706982),
  ('comptes', 0.11941674493706982)],
 3: [('challenge', 0.045760477078369714),
  ('innovation', 0.035843378360062135),
  ('energy', 0.032124930487526626),
  ('amp', 0.0298607417080666),
  ('entrepreneurship', 0.02909284263683155),
  ('entrepreneur', 0.0290588393789799),
  ('news', 0.025892079063791474),
  ('changemakers', 0.02381122173620606),
  ('enterprise', 0.02318702828461175),
  ('environmental', 0.022176112420584773)],
 4: [('nicki', 0.1313074865784844),
  ('minaj', 0.130360473543224),
  ('reactionsm', 0.08021857057654457),
  ('pb', 0.07531065658346635),
  ('dw', 0.07264490837091944),
  ('davos', 0.06825955765731091),
  ('bidens', 0.06377396799442915),
  ('newshour', 0.06295116956554998),
  ('daily', 0.06213950185155512),
  ('documentary', 0.05971346477607126)],
 5: [('china', 0.2180207519568891),
  ('pres', 0.10270270016996909),
  ('prosperity', 0.10189100058262364),
  ('risen', 0.07334896355640846),
  ('viewsas', 0.07059410566438593),
  ('influence', 0.06951862874860472),
  ('newshour', 0.059066152564348384),
  ('reactionsm', 0.05495751419785642),
  ('strength', 0.05437755379928426),
  ('bodycam', 0.053730502021585566)],
 6: [('robot', 0.2826774669665731),
  ('harvard', 0.2086959261618073),
  ('sophia', 0.20253853217117962),
  ('taliban', 0.1627665888005557),
  ('defeated', 0.1605311361132715),
  ('economist', 0.1406040948363385),
  ('jimmy', 0.12902433923169496),
  ('afghanistan', 0.1254973947949475),
  ('mba', 0.11191351072638743),
  ('tonight', 0.10902225322929028)],
 7: [('reactionsm', 0.04674616431531839),
  ('godsent', 0.043196801553882655),
  ('anxiety', 0.035926254238226124),
  ('solving', 0.026900681561688497),
  ('reactionsk', 0.02670099046274069),
  ('videoconfusion', 0.022326713315304916),
  ('foundation', 0.021619451469468386),
  ('official', 0.0198612975445199),
  ('live', 0.01901903468199964),
  ('paypal', 0.01870978154037444)],
 8: [('tillerson', 0.1345493816360408),
  ('trump', 0.102991574487498),
  ('balvin', 0.08400743878572767),
  ('nbc', 0.0507868658590287),
  ('tillersons', 0.04331911246158699),
  ('tragediestrademark', 0.041454508908193306),
  ('milltrademarks', 0.041454508908193306),
  ('meek', 0.04059320216433257),
  ('faceface', 0.035759277854559436),
  ('safari', 0.03361833374845251)],
 9: [('charm', 0.31959076767138783),
  ('thetop', 0.20501754996526378),
  ('naaba', 0.20360085215317164),
  ('consult', 0.20222986233021242),
  ('luck', 0.19243329868692446),
  ('infertility', 0.15691312704050528),
  ('spiritual', 0.15691312704050528),
  ('lord', 0.1566815351932402),
  ('gold', 0.15414141548997842),
  ('humanity', 0.14781900912239482)]}
  ```
### Outcome of Topic Merging

By applying this merging process, we reduce the number of topics and make the clustering results more interpretable. The final topics are broader and less redundant, providing a clearer understanding of the main themes in the data.


## 17. **Generating Topic Labels Based on Top Words**

Once the topics are clustered and merged, it's important to assign meaningful labels to each topic. This helps in interpreting the results in a more human-readable way. The labels can be generated using the top `n` words that characterize each topic.

### Overview of the Label Generation Process

The label generation process involves the following steps:

1. **Extract Top Words**: We first extract the top `n` words that represent each topic.
2. **Generate Labels**: Using a prompt, we generate labels that describe the content of each topic by incorporating these top words.
3. **Add Labels to Data**: The generated labels are then added to the dataset, making the topics more interpretable.

### Step 1: Generating Topic Labels

The function `generate_labels` is used to create a descriptive label for each topic. It uses the top `n` words of a topic and combines them into a string, which is then fed into a language model to generate a descriptive label.

```python
# Generate topic labels using the given prompts
def generate_labels(prompt, top_n_words):
    # Initialize an empty list to store labels
    labels = []
    # Iterate over each topic's top words
    for topic, words in top_n_words.items():
        # Join the words with commas
        word_str = ', '.join([word[0] for word in words])
        # Replace [DOCUMENTS] tag with the topic's top words
        prompt_with_words = prompt.replace("[KEYWORDS]", word_str)
        # Generate the label using the prompt
        label = generator(prompt_with_words)[0]['generated_text'].split('\n')[-1].strip()
        # Append the label to the list
        labels.append(label)
    return labels
```

### Explanation of the Code:
- **Iterating over each topic**: For each topic, we take its top `n` words and combine them into a string (separated by commas).
- **Creating the prompt**: A pre-defined prompt (usually a template) is used to guide the label generation. The `[KEYWORDS]` tag in the prompt is replaced with the top words for each topic.
- **Generating the label**: We use a text generation model (e.g., a language model) to generate a label that describes the topic based on the prompt.

### Step 2: Adding Generated Labels to the Dataset

Once the labels are generated, we add them to the dataset (`docs_per_topic`) for better interpretability.

```python
# Generate labels using the main prompt
labels = generate_labels(main_prompt, top_n_words)

# Add labels to the DataFrame
docs_per_topic['llama2_labelswsp'] = labels

# Display the DataFrame with the new labels
docs_per_topic
```

### Outcome:
- The dataset `docs_per_topic` now includes a new column, `llama2_labelswsp`, which contains the descriptive labels generated for each topic.
- These labels help in understanding the thematic content of each topic based on the most important words.

### Step 3: Saving the Labeled Dataset

Finally, we save the DataFrame with the topic labels to a new CSV file, so you can use it for further analysis or visualization.

```python
# Save the DataFrame to a new CSV file
docs_per_topic.to_csv('labeled_topics.csv', index=False)
```

By using this approach, each topic will have a descriptive label that makes the clustering results more interpretable. 

---

