# **YouTube Data Collection**
In this section, we describe how we collected YouTube data related to social concerns in Africa. The process involved using the YouTube Data API for structured data collection and Selenium for web scraping to extract additional details.

## 1. Introduction
We collected YouTube data using two main methods:

YouTube Data API: To fetch metadata such as video statistics, tags, and categories.
Web Scraping (Selenium): To gather video descriptions, titles, and links not available via the API.
2.
## 2. Summary of Collected Data

Below is a sample of the data collected using the YouTube Data API:
![Summary of Collected Data](../youtube_data.png)

## 2. Code for Collecting YouTube Data by API

We used the `youtube_search` function defined in the `youtube_data.py` file to search for videos related to social concerns. The function collects data such as video title, view count, comment count, etc. Here's the code used for this process:

### Importing Necessary Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from youtube_data import youtube_search
```

### Defining the `read_youtube` Function

This function takes a list of keywords, a start year, and an end year to search YouTube for videos related to social concerns.

```python
def read_youtube(list_title, start_year, end_year):
    tests = []
    for i in range(start_year, end_year + 1):
        for title in list_title:
            tests.append(youtube_search("social " + title + ", Africa, " + str(i)))
    return tests

test = read_youtube(["concern", "problem", "challenge", "worry", "issue", "question"], 2020, 2020)

df_test = []
for t in test:
    t.keys()
    df_test.append(pd.DataFrame(data=t))

df = pd.concat(df_test)
df.to_excel("concerns120202020.xlsx")
```

### Processing and Saving Data

We processed the data collected and saved it to both Excel and CSV formats for further analysis.

```python
df_test = []
for t in test:
    t.keys()
    df_test.append(pd.DataFrame(data=t))

df = pd.concat(df_test)
df.to_excel("concerns120202020.xlsx")
df.to_csv('concerns20202020.csv', sep=',', index=False)
```

### Code for Scraping YouTube Data Using Selenium

We used Selenium WebDriver to scrape video details such as title and description for each video. The following code demonstrates how we used Selenium to extract this information:

This guide explains how the provided Python script collects data from YouTube using Selenium and BeautifulSoup.

## Prerequisites
1. Install Python (preferably version 3.7 or higher).
2. Install the following Python libraries:
   - `selenium`
   - `pandas`
   - `beautifulsoup4`
   - `xlsxwriter`
3. Download the appropriate version of the ChromeDriver executable for your Chrome browser version.
4. Place the ChromeDriver executable in your working directory or update the path in the script.

---

## Step-by-Step Process

### 1. **Set Up the WebDriver**
- Import necessary libraries, including Selenium's `webdriver`, BeautifulSoup, and pandas.
- Configure ChromeDriver with options to disable unnecessary browser features:
  ```python
  driver_path = "chromedriver101.exe"
  option = Options()
  option.add_argument("--disable-infobars")
  option.add_argument("start-maximized")
  option.add_argument("--disable-extensions")
  driver = webdriver.Chrome(executable_path=driver_path, options=option)
  ```

### 2. **Access YouTube Search Results**
- Use the WebDriver to open YouTube's search results page for a specific query:
  ```python
  driver.get("https://www.youtube.com/results?search_query=social+questions%2C+Africa%2C+2022")
  ```

### 3. **Extract Video Links**
- Identify video elements on the page using XPath:
  ```python
  user_data = driver.find_elements(By.XPATH, '//*[@id="video-title"]')
  ```
- Collect links from the video elements:
  ```python
  links = [i.get_attribute('href') for i in user_data]
  ```

### 4. **Scroll and Load Additional Results**
- Implement a loop to scroll the page and dynamically load more results:
  ```python
  for timer in range(0, 1000):
      driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
      sleep(3)
  ```

### 5. **Parse Video Information**
- Use BeautifulSoup to parse the page source and locate relevant video metadata:
  ```python
  soup = BeautifulSoup(driver.page_source, "html.parser")
  all_posts = soup.find_all("ytd-video-renderer", {"class": "style-scope ytd-item-section-renderer"})
  ```

### 6. **Extract Metadata for Each Video**
- For each video post, extract details such as:
  - Title
  - Link
  - Description
  - Username
  - Time posted
  - Reactions
- Example for extracting the title:
  ```python
  try:
      title = post.find("a", {"class": "yt-simple-endpoint style-scope ytd-video-renderer"}).get('aria-label')
  except:
      title = "not found"
  ```

### 7. **Store Data in Lists**
- Append extracted data to corresponding lists for each attribute (e.g., `title_list`, `links_list`, etc.):
  ```python
  title_list.append(title)
  links_list.append(links)
  ```

### 8. **Save Data to Excel**
- Convert the collected data into a pandas DataFrame:
  ```python
  df = pd.DataFrame({
      "time": time_list,
      "links_ID": links_list,
      "username": username_list,
      "title": title_list,
      "desc1": desc1_list,
      "desc2": desc2_list,
      "reactions": reactions_list
  })
  ```
- Remove duplicates to avoid redundant entries:
  ```python
  df.drop_duplicates(subset="title", keep="first", inplace=True)
  ```
- Save the DataFrame to an Excel file:
  ```python
  df.to_excel("youquestions.xlsx")
  ```

### 9. **Terminate the Process**
- Break the loop once a sufficient amount of data (e.g., 1000 rows) is collected:
  ```python
  if df.shape[0] > 1000:
      break
  ```
