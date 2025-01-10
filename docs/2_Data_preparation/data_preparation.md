Got it! Here is the revised markdown with the sample output displaying all columns, including the ones created during the data cleaning process:

```markdown
# Data Preparation: Cleaning Twitter and YouTube Data

This document outlines the steps followed to clean Twitter and YouTube data and merge them for further analysis and machine learning model preparation.

## 1. Install Necessary Libraries

We begin by installing and importing the necessary libraries and packages required for text preprocessing, data manipulation, and machine learning:

```python
# Installing necessary libraries
!pip install numpy==1.21
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from dateutil import parser
import re
import langid
from deep_translator import GoogleTranslator
```

## 2. Load and Clean Data

We load both Twitter and YouTube datasets and perform the following cleaning steps:

### 2.1. Load the YouTube Data

```python
youtube = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/youtube.csv')
youtube = youtube.astype(str)
youtube['Texty'] = youtube[['title','desc1','desc2','reactions','time']].apply(lambda x: ''.join(x), axis=1)
```

### 2.2. Extract and Parse Dates for YouTube Data

We extract dates from the `Texty` column using regular expressions and date parsing techniques:

```python
def extract_date(row):
    text = row['Texty']
    # Extract date-related text and parse it to a datetime object
    # Implement logic for parsing relative time (e.g., "2 years ago", "3 days ago")
    ...
youtube['Date'] = youtube.apply(extract_date, axis=1)
```

### 2.3. Extract Views from YouTube Data

We extract the view count from the `Texty` column:

```python
def extract_views(text):
    match = re.search(r'(\d+(?:\.\d+)?(?:k)?) views', text, re.IGNORECASE)
    if match:
        views_text = match.group(1)
        if 'k' in views_text.lower():
            return int(float(views_text[:-1]) * 1000)
        else:
            return int(views_text.replace(',', ''))
    return 0

youtube['Views'] = youtube['Texty'].apply(extract_views)
```

### 2.4. Clean and Prepare the Twitter Data

Similarly, we clean the Twitter dataset by extracting relevant columns and calculating the view count based on interactions such as Retweets, Likes, and Replies:

```python
twitter = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/twitter.csv')
twitter['Date'] = twitter[['Datetime']]
twitter['Views'] = (twitter['RetweetCount'] + twitter['Likes'] + twitter['Replies']).astype(int).astype(str)
twitter = twitter[['Date', 'Views', 'Tweet Id', 'RetweetCount', 'Likes', 'Replies', 'Text', 'language', 'Username']]
```

### 2.5. Extract Views from Twitter Data

```python
twitterdf = twitter[['Text', 'Date', 'Views']]
twitterdf = twitterdf.dropna(how='all')
twitterdf['sourcetype'] = 'twitter'
```

### 2.6. Add Unique IDs for YouTube and Twitter Data

We add unique identifiers to the Twitter and YouTube datasets for future merging:

```python
youtube['yid'] = range(1, len(youtube) + 1)
twitterdf['tid'] = range(1, len(twitterdf) + 1)
```

## 3. Remove Duplicates and Missing Values

### 3.1. Remove Duplicates from YouTube Data

```python
newyoutube_df = youtube.drop_duplicates('Texty')
youtube2 = newyoutube_df[["yid", "Date", "Texty", "sourcetype", "Views"]]
```

### 3.2. Remove Duplicates from Twitter Data

```python
twitter2 = twitterdf.drop_duplicates('Text')
```

### 3.3. Clean the DataFrames

We now clean the datasets by dropping unnecessary columns and removing any rows with missing values:

```python
dfs = [twitter2, youtube2]
df = pd.concat(dfs, axis=0)
df = df.dropna()  # Drop rows with missing values
```

## 4. Combine Text Columns

We combine the `Text`, `Textf`, and `Texty` columns to create a unified text column for analysis:

```python
df[['Text', 'textf', 'Texty']] = df[['Text', 'textf', 'Texty']].fillna('')
df['alltext'] = df['Text'] + ' ' + df['textf'] + ' ' + df['Texty']
df['textid'] = range(1, len(df) + 1)
```

## 5. Convert Emoticons and Emojis

### 5.1. Remove Emoticons

We define a function to remove emoticons:

```python
def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

df['Text_emt'] = df['alltext'].apply(remove_emoticons)
```

### 5.2. Convert Emoticons and Emojis to Words

We convert emoticons and emojis into readable text using predefined dictionaries:

```python
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = re.sub(r'('+emot+')', "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()), text)
    return text

df['Text_emj'] = df['Text_emt'].apply(convert_emojis)
```

## 6. Language Detection and Translation

We detect and translate text that is not in English using the `langdetect` library and `GoogleTranslator`:

```python
from langdetect import detect
def detect_my(text):
    try:
        return detect(text)
    except:
        return 'unknown'

df['Text_langue'] = df['Text_emj'].apply(detect_my)
```

Below is the distribution of detected languages in the dataset:
![Language Detection and Translation](../language_detection.png)


```python
gt = GoogleTranslator(source='auto', target='en')
translated_arts = []
for i in range(len(train_not_en)):
    cur_art = train_not_en.iloc[i]['Text_emj'][:1000]
    translated_arts.append(cur_art if langid.classify(cur_art)[0] == 'en' else gt.translate(cur_art))
train_not_en['Text_translate'] = translated_arts
```

## 7. Additional Text Cleaning

### 7.1. Remove Extra Whitespaces

```python
def remove_extra_whitespaces(text):
    return " ".join(text.split())

df['Text_cleaned'] = df['Text_lemma'].apply(remove_extra_whitespaces)
```

### 7.2. Remove Non-ASCII Characters

```python
def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

df['Text_cleaned'] = df['Text_cleaned'].apply(remove_non_ascii)
```

### 7.3. Remove Short Words

```python
def remove_short_words(text, min_length=3):
    return ' '.join([word for word in text.split() if len(word) >= min_length])

df['Text_cleaned'] = df['Text_cleaned'].apply(remove_short_words)
```

### 7.4. Handle Missing Data

**Fill Missing Data**: Replace `NaN` values with a placeholder (e.g., 'missing') if there are any missing values in the cleaned text.

```python
df['Text_cleaned'] = df['Text_cleaned'].fillna('missing')
```

### 7.5. Tokenization

```python
from nltk.tokenize import word_tokenize

def tokenizingText(text):
    text = word_tokenize(text)
    return text

df['Text_token'] = df['Text_cleaned'].apply(tokenizingText)
```

### 7.6. Remove Stopwords

```python
def stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords.words('english')])

df['Text_stop'] = df['Text_cleaned'].apply(stopwords)
```

### 7.7. Stemming

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

df['Text_stem'] = df['Text_stop'].apply(stem_words)
```

### 7.8. Lemmatization

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
   

 return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df["Text_lemma"] = df["Text_stop"].apply(lemmatize_words)
```

## 8. Handling Data with Missing Values

### 8.1. Check for Missing Data

```python
print("No. of columns containing null values")
print(len(df.columns[df.isna().any()]))

print("No. of columns not containing null values")
print(len(df.columns[df.notna().all()]))
```

### 8.2. Drop Rows with Missing Values

```python
df = df.dropna()
```

## 9. Save the Final Data to CSV

```python
df.to_csv('/content/drive/MyDrive/Colab Notebooks/socialcons_clean2.csv')
```

## 10. Sample Output

Below is a sample of the final output after all the data cleaning steps, showing the cleaned and processed columns:

```
  	textid	alltext	sourcetype	Text_stop	Text_stem	Text_lemma
11203	21534	Inmate kills cellmate and hides body without...	youtube	inmate kills cellmate hides body without guard...	inmat kill cellmat hide bodi without guard not...	inmate kill cellmate hide body without guard n...
15362	24868	was it a SET UP? : Rapper AKA's last post ra...	youtube	set rapper akas last post raises questions dur...	set rapper aka last post rais question durban ...	set rapper akas last post raise question durba...
8785	8583	Gee; why should we worry about someone whose f...	twitter	gee worry someone whose father moved family so...	gee worri someon whose father move famili sout...	gee worry someone whose father moved family so...
1834	1812	We can't wait to hear @Cc_HUB's plans for the ...	twitter	cant wait hear hubs plans launch amp global so...	cant wait hear hub plan launch amp global soci...	cant wait hear hub plan launch amp global soci...
1199	1190	Social inclusion is central to #Africaâ€™s ach...	twitter	social inclusion central trademarks achievemen...	social inclus central trademark achiev tackl c...	social inclusion central trademark achievement..
```
