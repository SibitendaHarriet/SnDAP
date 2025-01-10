# **Twitter Data Collection**

This guide explains how to collect Twitter data without requiring a Twitter Developer Account using the `snscrape` tool.
```markdown
## 1. Introduction

`snscrape` is a command-line and Python library for scraping tweets. Unlike Twitter's API, it does not require developer credentials or an API key, making it accessible and easy to use for researchers.
```
## 2. Installation

Follow these steps to install `snscrape`:

### Requirements:
- Python version: **3.8 or higher**

### Install via pip:
```bash
pip install git+https://github.com/JustAnotherArchivist/snscrape.git
```
```
## 3. Usage Options

`snscrape` can be used in two ways:

1. **Command-Line Interface (CLI)**.
2. **Python Wrapper**.

### 3.1 Using the CLI

You can collect tweets directly using the command line. Run the following command:

```bash
snscrape --max-results 100 twitter-search "covid-19 lang:en until:2022-12-31" > tweets.json
```

**Explanation of Parameters**:
- `--max-results 100`: Limits the number of tweets collected to 100.
- `twitter-search`: Specifies the search query for Twitter data.
- `"covid-19 lang:en until:2022-12-31"`: Searches for English-language tweets containing "covid-19" and posted before December 31, 2022.
- `> tweets.json`: Saves the collected tweets to a file named `tweets.json`.

---

### 3.2 Using the Python Wrapper

For more control and flexibility, you can use the Python API provided by `snscrape`. Below is an example script:

```python
import snscrape.modules.twitter as sntwitter

# Define the search query
query = "covid-19 lang:en since:2020-12-01 until:2022-12-31"
max_tweets = 100

# Initialize an empty list to store tweets
tweets = []

# Use the TwitterSearchScraper to fetch tweets
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i >= max_tweets:  # Stop when max_tweets is reached
        break
    tweets.append([tweet.id, tweet.date, tweet.content])  # Save tweet data as a list

# Display the first tweet for verification
print(tweets[0])
```

**Explanation of the Code**:
1. `query`: Defines the search criteria for tweets, including language (`lang:en`) and the date range (`since` and `until`).
2. `max_tweets`: Sets the maximum number of tweets to retrieve.
3. `TwitterSearchScraper`: A method from `snscrape` that fetches tweets based on the query.
4. `tweets.append`: Stores the tweet ID, date, and content in a list.
5. `print(tweets[0])`: Displays the first tweet in the collected data.

---

## 4. Saving Data to a CSV File

Once you've collected the tweets, it’s useful to save them in a structured format like CSV for further analysis. Here’s how to do it:

```python
import pandas as pd

# Convert the list of tweets to a DataFrame
df = pd.DataFrame(tweets, columns=["ID", "Date", "Content"])

# Save the DataFrame to a CSV file
df.to_csv("tweets.csv", index=False)
```

**Explanation of the Code**:
1. `pd.DataFrame`: Converts the collected tweets into a tabular format with columns for tweet ID, date, and content.
2. `to_csv`: Saves the DataFrame to a CSV file named `tweets.csv`.
3. `index=False`: Prevents the addition of an unnecessary index column in the CSV file.

---

## 5. Next Steps

After collecting and saving the tweets, you can perform various analyses, such as:

- **Sentiment Analysis**: Use Natural Language Processing (NLP) techniques to analyze the sentiment of tweets.
- **Topic Modeling**: Apply algorithms like Latent Dirichlet Allocation (LDA) to identify common themes.
- **Network Analysis**: Explore relationships between users or hashtags.

These analyses can help you derive insights from the data you collected.

---

## 6. Summary

In this guide, we covered:

1. Installation and setup of `snscrape`.
2. Using both the CLI and Python wrapper to scrape tweets.
3. Saving the collected data to a CSV file for further analysis.

With this knowledge, you are now ready to collect and analyze Twitter data!
```

