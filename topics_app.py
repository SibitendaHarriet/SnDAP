import numpy as np
import pandas as pd
import joblib
import umap
import streamlit as st
from sklearn import preprocessing
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer
# from keras_preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence
from prophet import Prophet
import networkx as nx
# import community as community_louvain  # For Louvain method
import torch.nn.functional as F
from community import community_louvain
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.decomposition import PCA

# from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
import pyautogui
import os

# Create a directory to store screenshots
os.makedirs("screenshots", exist_ok=True)

# Streamlit app
st.header("Social Network Data Analysis and Prediction App")

# Load the already trained data
trained_data_path = './data/train.csv'
trained_df = pd.read_csv(trained_data_path, nrows=5000)  # Load only 5000 rows
trained_df = trained_df.astype(str)

# Sidebar for navigation
# page = st.sidebar.selectbox("Choose a page", ["Topics", "Themes", "Sentiment"])
page = st.sidebar.selectbox("Choose a page", ["Topics", "Themes", "Sentiment", "Temporal Analysis", "Social Network Analysis"])


# Load models and set up tokenizer for new predictions
MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
embedding_model = SentenceTransformer(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set up MultiLabelBinarizer for the six specific themes only
selected_themes = ['poverty', 'education', 'hunger', 'security', 'employment', 'health']
mlb = MultiLabelBinarizer(classes=selected_themes)
mlb.fit([selected_themes])  # Fit the binarizer on the six themes without involving cleaned_theme


# Function to clean topic text
def extract_topic(text):
    if text is None:
        return None
    exclusion_phrases = [
        "As a responsible", "It is important", "Therefore", "Remember", "I hope you understand", 
        "I cannot provide a", "I apologize", "Please provide me with more", 
        "If you have any other", "Instead", "It's important to"
    ]
    if any(phrase in text for phrase in exclusion_phrases):
        return None
    if text.startswith(('Sure!', 'Based on the keywords provided', 'Certainly! Based on the keywords provided')):
        # Regular expression to extract text within quotes
        topic_match = re.search(r'"(.*?)"', text)
        if topic_match:
            return topic_match.group(1)
        else:
            return text
    else:
        return text

# --- Topics Page ---
if page == "Topics":
    st.subheader("Topics Analysis")

    # 1. User Interaction with Trained Data
    st.subheader("Word Cloud for Topics")
    unique_words = set(' '.join(trained_df['llama2_labelone'].astype(str).str.lower().str.strip()).split())
    cleaned_text = ' '.join(unique_words)
    wordcloud_topics = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
    fig1, ax1 = plt.subplots()
    ax1.imshow(wordcloud_topics, interpolation='bilinear')
    ax1.axis('off')
    st.pyplot(fig1)

    # Save Word Cloud Image and Screenshot
    fig1.savefig("wordcloud_topics.png")
    st.write("Word Cloud saved as 'wordcloud_topics.png'.")
    pyautogui.screenshot("screenshots/wordcloud_topics_screenshot.png")
    st.write("Screenshot saved as 'wordcloud_topics_screenshot.png'.")

    # Histogram for Topics
    st.subheader("Histogram for Topics")
    fig2 = px.histogram(trained_df, x='llama2_labelone', title='Common Topics in llama2_labelone')
    st.plotly_chart(fig2)

    # Save Plotly Histogram and Screenshot
    fig2.write_html("histogram_topics.html")
    st.write("Histogram saved as 'histogram_topics.html'.")
    pyautogui.screenshot("screenshots/histogram_topics_screenshot.png")
    st.write("Screenshot saved as 'histogram_topics_screenshot.png'.")

    # Display Topics with Adjustable Count Range
    st.subheader("Topics with Adjustable Count Range")
    min_count, max_count = st.slider("Select range for topic counts", min_value=1, max_value=100, value=(20, 50))
    filtered_topics = trained_df['llama2_labelone'].value_counts()
    filtered_topics = filtered_topics[(filtered_topics >= min_count) & (filtered_topics <= max_count)].head(100)

    if not filtered_topics.empty:
        fig3, ax3 = plt.subplots()
        sns.barplot(x=filtered_topics.values, y=filtered_topics.index, palette="viridis", ax=ax3)
        for j, value in enumerate(filtered_topics.values):
            percentage = f"{(value / len(trained_df)) * 100:.1f}%"
            ax3.text(value + 0.5, j, percentage, ha='left', va='center')
        ax3.set_title(f"Topics with Counts Between {min_count} and {max_count} (Max 100 Topics)")
        ax3.set_xlabel("Count")
        ax3.set_ylabel("Topic")
        st.pyplot(fig3)

        # Save Bar Plot and Screenshot
        fig3.savefig("filtered_topics_barplot.png")
        st.write("Filtered Topics Bar Plot saved as 'filtered_topics_barplot.png'.")
        pyautogui.screenshot("screenshots/filtered_topics_screenshot.png")
        st.write("Screenshot saved as 'filtered_topics_screenshot.png'.")
    else:
        st.write(f"No topics found within the count range {min_count} to {max_count}.")

    # 2. File Upload for Predictions on Test Dataset
    uploaded_file = st.file_uploader("Upload CSV for Topic Predictions", type="csv")
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        test_df = test_df.astype(str)

        documents = [' '.join(doc.split()[:MAX_LEN]) for doc in test_df['Text_lemma'].values]
        embeddings = embedding_model.encode(documents, convert_to_tensor=True).cpu().numpy()
        X_embedded = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)
        x_new = preprocessing.scale(X_embedded)

        model_topics = joblib.load('./model/stack_classifier_modelwsp2.pkl')
        predictions_topics = model_topics.predict(x_new)
        
        # Use extract_topic to clean each prediction
        test_df['Predicted_Topics'] = [str(label) for label in predictions_topics]
        test_df['Cleaned_Predicted_Topics'] = test_df['Predicted_Topics'].apply(extract_topic)

        # Display and Download Predictions
        st.subheader("Predicted Topics for Uploaded Data")
        st.write(test_df[['Text_lemma', 'Predicted_Topics', 'Cleaned_Predicted_Topics']])
        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predicted Topics as CSV", csv, "predicted_topics.csv", "text/csv")

        # Take Screenshot of Prediction Table
        pyautogui.screenshot("screenshots/predicted_topics_table_screenshot.png")
        st.write("Screenshot of prediction table saved as 'predicted_topics_table_screenshot.png'.")

# --- Themes Page ---
elif page == "Themes":
    st.subheader("Themes Analysis")

    # 1. Topic-Based Themes Display from 'cleaned_theme'
    st.subheader("Word Cloud for Themes")
    unique_words = set(' '.join(trained_df['cleaned_theme'].astype(str).str.lower().str.strip()).split())
    cleaned_text = ' '.join(unique_words)
    wordcloud_themes = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
    fig4, ax4 = plt.subplots()
    ax4.imshow(wordcloud_themes, interpolation='bilinear')
    ax4.axis('off')
    st.pyplot(fig4)

    # Save Word Cloud Image and Screenshot
    fig4.savefig("wordcloud_themes.png")
    st.write("Word Cloud saved as 'wordcloud_themes.png'.")
    pyautogui.screenshot("screenshots/wordcloud_themes_screenshot.png")
    st.write("Screenshot saved as 'wordcloud_themes_screenshot.png'.")

    # Topic-Based Histogram for Themes
    st.subheader("Histogram for Themes")
    fig5 = px.histogram(trained_df, x='cleaned_theme', title='Common Themes in Social Comments')
    st.plotly_chart(fig5)

    # Save Plotly Histogram and Screenshot
    fig5.write_html("histogram_themes.html")
    st.write("Histogram saved as 'histogram_themes.html'.")
    pyautogui.screenshot("screenshots/histogram_themes_screenshot.png")
    st.write("Screenshot saved as 'histogram_themes_screenshot.png'.")

    # Display Themes Based on Count Range (Topic-Based Themes)
    st.subheader("Themes within a Count Range")
    min_count, max_count = st.slider("Select count range", 2, 100, (2, 50))
    filtered_themes = trained_df['cleaned_theme'].value_counts()
    filtered_themes = filtered_themes[(filtered_themes >= min_count) & (filtered_themes <= max_count)]

    if not filtered_themes.empty:
        fig5, ax4 = plt.subplots()
        sns.barplot(x=filtered_themes.values, y=filtered_themes.index, palette="viridis", ax=ax4)
        for j, value in enumerate(filtered_themes.values):
            percentage = f"{(value / len(trained_df)) * 100:.1f}%"
            ax4.text(value + 0.5, j, percentage, ha='left', va='center')
        ax4.set_title(f"Themes with Counts Between {min_count} and {max_count}")
        ax4.set_xlabel("Count")
        ax4.set_ylabel("Theme")
        st.pyplot(fig5)

        # Save Bar Plot and Screenshot
        fig5.savefig("filtered_themes_barplot.png")
        st.write("Filtered Themes Bar Plot saved as 'filtered_themes_barplot.png'.")
        pyautogui.screenshot("screenshots/filtered_themes_screenshot.png")
        st.write("Screenshot saved as 'filtered_themes_screenshot.png'.")
    else:
        st.write(f"No themes found within the count range {min_count} to {max_count}.")

    # 2. File Upload for Predictions on Test Dataset
    uploaded_file = st.file_uploader("Upload CSV for Theme Predictions", type="csv")
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        test_df = test_df.astype(str)

        documents = [' '.join(doc.split()[:MAX_LEN]) for doc in test_df['Text_lemma'].values]
        embeddings = embedding_model.encode(documents, convert_to_tensor=True).cpu().numpy()
        X_embedded = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)
        x_new = preprocessing.scale(X_embedded)

        # Single-label prediction for comparison
        model_themes = joblib.load('./model/stack_classifier_modelwsp2_theme.pkl')
        predictions_themes = model_themes.predict(x_new)
        test_df['Predicted_Themes'] = predictions_themes

        # Multi-label prediction for the six themes
        model_themes2 = joblib.load('./model/stack_classifier_modeltheme.pkl')
        predictions_themes2 = model_themes2.predict(x_new)

        # Decode multi-label predictions using mlb.inverse_transform
        predicted_labels = mlb.inverse_transform(predictions_themes2)
        test_df['Predicted_Categories3'] = predicted_labels

        # Display and Download Predictions
        st.subheader("Predicted Themes for Uploaded Data")
        st.write(test_df[['Text_lemma', 'Predicted_Themes', 'Predicted_Categories3']])
        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predicted Themes as CSV", csv, "predicted_themes.csv", "text/csv")

        # Take Screenshot of Prediction Table
        pyautogui.screenshot("screenshots/predicted_themes_table_screenshot.png")
        st.write("Screenshot of prediction table saved as 'predicted_themes_table_screenshot.png'.")

        # 3. Display Bar Graph for Predicted Themes Count
        st.subheader("Predicted Themes Count Distribution")
        pthemes_counts = test_df['Predicted_Themes'].value_counts()

        # Display Themes with Adjustable Count Range
        st.subheader("Predicted_Themes with Adjustable Count Range")
        min_count, max_count = st.slider("Select range for predicted themes counts", min_value=1, max_value=100, value=(20, 50))
        filtered_pthemes = test_df['Predicted_Themes'].value_counts()
        filtered_pthemes = filtered_pthemes[(filtered_pthemes >= min_count) & (filtered_pthemes <= max_count)].head(100)

        # Plot bar chart using Plotly for interactivity
        fig4 = px.bar(
            pthemes_counts, 
            x=pthemes_counts.index, 
            y=pthemes_counts.values, 
            title="Distribution of Predicted Themes",
            labels={'x': 'Predicted_Themes', 'y': 'Count'}
        )
        fig4.update_layout(xaxis_title="Predicted_Themes", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig4)

        # Save Predicted Themes Bar Plot and Screenshot
        fig4.write_html("predicted_themes_barplot.html")
        st.write("Predicted Themes Bar Plot saved as 'predicted_themes_barplot.html'.")
        pyautogui.screenshot("screenshots/predicted_themes_barplot_screenshot.png")
        st.write("Screenshot of Predicted Themes Bar Plot saved as 'predicted_themes_barplot_screenshot.png'.")

        # Interactive Multi-label Count Display for Each Theme
        st.subheader("Counts of Each Multi-label Theme")
        theme_counts = {theme: sum(test_df['Predicted_Categories3'].apply(lambda x: theme in x)) for theme in selected_themes}
        
        # Plot the counts in an interactive bar chart using Plotly
        fig6 = px.bar(
            x=list(theme_counts.keys()),
            y=list(theme_counts.values()),
            title="Count of Each Theme in Multi-label Predictions",
            labels={'x': 'Theme', 'y': 'Count'},
            color_discrete_sequence=['#636EFA']
        )
        fig6.update_layout(xaxis_title="Theme", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig6)

        # Save Multi-label Theme Count Bar Plot and Screenshot
        fig6.write_html("multi_label_theme_counts.html")
        st.write("Multi-label Theme Count Bar Plot saved as 'multi_label_theme_counts.html'.")
        pyautogui.screenshot("screenshots/multi_label_theme_counts_screenshot.png")
        st.write("Screenshot of Multi-label Theme Count Bar Plot saved as 'multi_label_theme_counts_screenshot.png'.")


# --- Sentiment Page ---
elif page == "Sentiment":
    st.subheader("Sentiment Analysis")

    # Dictionary for mapping numeric sentiment labels to text
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    # 1. User Interaction with Trained Data
    st.subheader("Sentiment Distribution in Comments")
    sentiment_counts = trained_df['llama3_sentiment'].value_counts()
    total_comments = len(trained_df)

    # Function to plot distribution with percentage annotations
    def plot_distribution(counts, title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)
        for i, value in enumerate(counts.values):
            percentage = f"{(value / total_comments) * 100:.1f}%"
            ax.text(i, value + 5, percentage, ha='center', va='bottom')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig

    # Plot sentiment distribution
    fig6 = plot_distribution(sentiment_counts, "Sentiment Distribution", "Sentiment", "Count")
    st.pyplot(fig6)

    # Save Sentiment Distribution Bar Plot and Screenshot
    fig6.savefig("sentiment_distribution.png")
    st.write("Sentiment Distribution saved as 'sentiment_distribution.png'.")
    pyautogui.screenshot("screenshots/sentiment_distribution_screenshot.png")
    st.write("Screenshot saved as 'sentiment_distribution_screenshot.png'.")

    # Top Topics for Each Sentiment
    st.subheader("Top Topics for Each Sentiment")
    selected_sentiment = st.selectbox("Select a sentiment to view top topics", options=sentiment_counts.index)
    sentiment_df = trained_df[trained_df['llama3_sentiment'] == selected_sentiment]
    top_topics = sentiment_df['llama2_labelone'].value_counts().head(10)

    fig7, ax6 = plt.subplots()
    sns.barplot(x=top_topics.values, y=top_topics.index, palette="viridis", ax=ax6)
    for j, value in enumerate(top_topics.values):
        percentage = f"{(value / len(sentiment_df)) * 100:.1f}%"
        ax6.text(value + 0.5, j, percentage, ha='left', va='center')
    ax6.set_title(f"Top Topics for '{selected_sentiment}' Sentiment")
    ax6.set_xlabel("Count")
    ax6.set_ylabel("Topics")
    st.pyplot(fig7)

    # Save Top Topics for Sentiment Bar Plot and Screenshot
    fig7.savefig(f"top_topics_{selected_sentiment}_sentiment.png")
    st.write(f"Top Topics for '{selected_sentiment}' Sentiment saved as '{selected_sentiment}_sentiment.png'.")
    pyautogui.screenshot(f"screenshots/top_topics_{selected_sentiment}_sentiment_screenshot.png")
    st.write(f"Screenshot saved as 'top_topics_{selected_sentiment}_sentiment_screenshot.png'.")

    # Top Themes for Each Sentiment
    st.subheader("Top Themes for Each Sentiment")
    top_themes = sentiment_df['cleaned_theme'].value_counts().head(10)
    fig8, ax7 = plt.subplots()
    sns.barplot(x=top_themes.values, y=top_themes.index, palette="viridis", ax=ax7)
    for j, value in enumerate(top_themes.values):
        percentage = f"{(value / len(sentiment_df)) * 100:.1f}%"
        ax7.text(value + 0.5, j, percentage, ha='left', va='center')
    ax7.set_title(f"Top Themes for '{selected_sentiment}' Sentiment")
    ax7.set_xlabel("Count")
    ax7.set_ylabel("Theme")
    st.pyplot(fig8)

    # Save Top Themes for Sentiment Bar Plot and Screenshot
    fig8.savefig(f"top_themes_{selected_sentiment}_sentiment.png")
    st.write(f"Top Themes for '{selected_sentiment}' Sentiment saved as '{selected_sentiment}_sentiment.png'.")
    pyautogui.screenshot(f"screenshots/top_themes_{selected_sentiment}_sentiment_screenshot.png")
    st.write(f"Screenshot saved as 'top_themes_{selected_sentiment}_sentiment_screenshot.png'.")

    # 2. File Upload for Sentiment Predictions
    uploaded_file = st.file_uploader("Upload CSV for Sentiment Predictions", type="csv")
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file).astype(str)

        # Load the saved model and tokenizer
        output_dir = "./model/saved_model/"
        model = BertForSequenceClassification.from_pretrained(output_dir)
        tokenizer = BertTokenizer.from_pretrained(output_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set the maximum sequence length
        MAX_LEN = 128  # Should match the max length used during training

        # Assuming you have a column 'Text_lemma' in your new dataset
        new_text_data = test_df['Text_lemma']

        # Tokenize the text data using the BERT tokenizer
        new_tokenized_texts = [tokenizer.tokenize(sent) for sent in new_text_data]

        # Convert tokens to index numbers and create attention masks
        new_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in new_tokenized_texts]
        new_input_ids = [torch.tensor(seq[:MAX_LEN], dtype=torch.long) for seq in new_input_ids]
        new_input_ids_padded = pad_sequence(new_input_ids, batch_first=True, padding_value=0)
        new_prediction_inputs = new_input_ids_padded  # Already padded tensor

        # Create attention masks directly as a tensor (1 where non-zero values, 0 otherwise)
        new_attention_masks = torch.where(new_prediction_inputs != 0, torch.tensor(1), torch.tensor(0))
        new_prediction_masks = new_attention_masks  # This is already calculated as a tensor

        # Set the model to evaluation mode
        model.eval()

        # Create DataLoader for the new dataset
        new_prediction_data = TensorDataset(new_prediction_inputs, new_prediction_masks)
        new_prediction_sampler = SequentialSampler(new_prediction_data)
        new_prediction_dataloader = DataLoader(new_prediction_data, sampler=new_prediction_sampler, batch_size=32)  # Match batch size to training

        # Tracking variables for predictions
        new_predictions = []

        # Predict
        for batch in new_prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits.detach().cpu().numpy()  # Extract logits and convert to numpy
            new_predictions.append(logits)

        # Aggregate and analyze the predictions for the new dataset
        flat_new_predictions = [item for sublist in new_predictions for item in sublist]
        flat_new_predictions = np.argmax(flat_new_predictions, axis=1).flatten()

        # Add numeric predictions to the new dataset and map them to text labels
        test_df['Predicted_Sentiment'] = flat_new_predictions
        test_df['Predicted_Sentiment_Text'] = test_df['Predicted_Sentiment'].map(sentiment_mapping)

        # Display and Download Predictions
        st.subheader("Predicted Sentiment for Uploaded Data")
        st.write(test_df[['Text_lemma', 'Predicted_Sentiment_Text']])
        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predicted Sentiment as CSV", csv, "predicted_sentiment.csv", "text/csv")

        # 2. Display Bar Graph for Sentiment Counts
        st.subheader("Predicted Sentiment Count Distribution")
        sentiment_counts = test_df['Predicted_Sentiment_Text'].value_counts()

        fig9, ax9 = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", ax=ax9)
        for j, value in enumerate(sentiment_counts.values):
            percentage = f"{(value / len(test_df)) * 100:.1f}%"
            ax9.text(j, value + 0.5, percentage, ha='center', va='bottom')
        ax9.set_title("Predicted Sentiment Distribution")
        ax9.set_xlabel("Sentiment")
        ax9.set_ylabel("Count")
        st.pyplot(fig9)

        # Save Predicted Sentiment Bar Plot and Screenshot
        fig9.savefig("predicted_sentiment_distribution.png")
        st.write("Predicted Sentiment Distribution saved as 'predicted_sentiment_distribution.png'.")
        pyautogui.screenshot("screenshots/predicted_sentiment_distribution_screenshot.png")
        st.write("Screenshot saved as 'predicted_sentiment_distribution_screenshot.png'.")


elif page == "Temporal Analysis":
    st.title("Temporal Analysis of Themes and Topics Over Time")

    if 'Date' in trained_df.columns:
        trained_df['Date'] = pd.to_datetime(trained_df['Date'], errors='coerce')
        trained_df = trained_df.dropna(subset=['Date'])  # Remove rows with invalid dates

        # Dropdown for selecting analysis type
        temporal_category = st.selectbox("Select category for temporal analysis", ["Topics", "Themes"])

        # Dropdown for time granularity
        time_granularity = st.selectbox("Select time granularity", ["Daily", "Weekly", "Monthly", "Yearly"])

        # Set grouping frequency based on granularity
        date_freq = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Yearly': 'Y'}[time_granularity]

        # Select sentiment
        sentiment_category = st.selectbox("Select sentiment for observation", ["All Sentiments", "Positive", "Negative", "Neutral"])

        # Determine the column to analyze based on user selection
        category_column = 'llama2_labelone' if temporal_category == "Topics" else 'cleaned_theme'

        # Frequency range slider to filter topics or themes
        category_counts = trained_df[category_column].value_counts()
        min_count, max_count = category_counts.min(), category_counts.max()
        
        # Allow user to select a range for frequency filtering
        min_freq, max_freq = st.slider(
            f"Select frequency range for filtering {temporal_category.lower()}",
            min_value=int(min_count),
            max_value=int(max_count),
            value=(100, 500)  # Default range
        )

        # Filter categories within the selected frequency range
        filtered_categories = category_counts[(category_counts >= min_freq) & (category_counts <= max_freq)].index
        filtered_df = trained_df[trained_df[category_column].isin(filtered_categories)]

        # Further filter by sentiment if not 'All Sentiments'
        if sentiment_category != "All Sentiments":
            filtered_df = filtered_df[filtered_df['llama3_sentiment'] == sentiment_category]

        # Group by category, date frequency, and sentiment
        aggr_df = filtered_df.groupby([category_column, pd.Grouper(key='Date', freq=date_freq), 'llama3_sentiment'])['Text_lemma'].count().reset_index()
        aggr_df.columns = [category_column, 'Date', 'llama3_sentiment', 'Posts']

        # Create plots for each sentiment
        unique_sentiments = aggr_df['llama3_sentiment'].unique()
        fig, axes = plt.subplots(nrows=len(unique_sentiments), ncols=1, figsize=(10, len(unique_sentiments) * 4), sharex=True)

        if len(unique_sentiments) == 1:
            axes = [axes]

        for i, sentiment in enumerate(unique_sentiments):
            ax = axes[i]
            sentiment_data = aggr_df[aggr_df['llama3_sentiment'] == sentiment]
            for category in filtered_categories:
                category_data = sentiment_data[sentiment_data[category_column] == category]
                if not category_data.empty:
                    # Plot each category data and ensure label is added
                    ax.plot(category_data['Date'], category_data['Posts'], label=category, marker='o')
            ax.set_title(f'{sentiment.capitalize()} Sentiment')
            ax.set_ylabel('Number of Posts')
            # Adjust legend to appear below each subplot
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=3)

        fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        st.pyplot(fig)
        
        # Forecasting with Prophet for all selected categories
        if st.button("Forecast for Selected Categories"):
            forecast_data = []

            # Prepare data for each category to be forecasted
            for category in filtered_categories:
                category_df = aggr_df[aggr_df[category_column] == category][['Date', 'Posts']].copy()
                category_df.columns = ['ds', 'y']  # Rename columns for Prophet

                # Remove timezone if present
                category_df['ds'] = category_df['ds'].dt.tz_localize(None)

                # Initialize and fit the Prophet model
                model = Prophet(yearly_seasonality=True, daily_seasonality=False)
                model.fit(category_df)

                # Create a dataframe to hold future dates for predictions (360 days for one year)
                future = model.make_future_dataframe(periods=360)  # Forecast for 360 days
                forecast = model.predict(future)

                # Append category and forecast data to the list
                forecast['Category'] = category
                forecast_data.append(forecast)

            # Concatenate all forecasts into a single DataFrame
            all_forecasts = pd.concat(forecast_data)

            # Remove timezone from forecast data if present
            all_forecasts['ds'] = all_forecasts['ds'].dt.tz_localize(None)

            # Create subplots for forecasts based on sentiment
            fig_forecast, axes_forecast = plt.subplots(nrows=len(unique_sentiments), ncols=1, figsize=(10, len(unique_sentiments) * 4), sharex=True)

            if len(unique_sentiments) == 1:
                axes_forecast = [axes_forecast]

            for i, sentiment in enumerate(unique_sentiments):
                # Filter the forecasts based on sentiment
                sentiment_forecast = all_forecasts[all_forecasts['Category'].isin(filtered_categories)]

                # Filter the original data based on sentiment for plotting the historical data
                sentiment_data = filtered_df[filtered_df['llama3_sentiment'] == sentiment]
                sentiment_category_counts = sentiment_data.groupby([pd.Grouper(key='Date'), category_column]).size().reset_index(name='Posts')

                # Plot historical data for the sentiment
                for category in filtered_categories:
                    category_hist_data = sentiment_category_counts[sentiment_category_counts[category_column] == category]
                    if not category_hist_data.empty:
                        axes_forecast[i].plot(category_hist_data['Date'], category_hist_data['Posts'], label=f'{category} Historical', linestyle='-', marker='o')

                    # Get corresponding forecast for the category
                    category_forecast = sentiment_forecast[sentiment_forecast['Category'] == category]
                    if not category_forecast.empty:
                        axes_forecast[i].plot(category_forecast['ds'], category_forecast['yhat'], label=f'{category} Forecast', linestyle='--')
                        axes_forecast[i].fill_between(category_forecast['ds'], category_forecast['yhat_lower'], category_forecast['yhat_upper'], alpha=0.2)

                axes_forecast[i].set_title(f'Forecasts for {sentiment.capitalize()} Sentiment')
                axes_forecast[i].set_ylabel('Forecasted Posts')
                # Adjust legend to appear below each forecast subplot
                axes_forecast[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=3)

            fig_forecast.text(0.5, 0.04, 'Date', ha='center', fontsize=12)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            st.pyplot(fig_forecast)

            # Save Forecast Plot and Screenshot
            fig_forecast.savefig("forecast_sentiment_analysis.png")
            st.write("Forecast Sentiment Analysis plot saved as 'forecast_sentiment_analysis.png'.")
            pyautogui.screenshot("screenshots/forecast_sentiment_analysis_screenshot.png")
            st.write("Screenshot saved as 'forecast_sentiment_analysis_screenshot.png'.")

    # else:
        # st.warning("The dataset does not contain a 'Date' column.")

    else:
        st.write("The dataset does not contain a 'Date' column. Please upload data with date information to perform temporal analysis.")

# --- Social Network Analysis Page ---
elif page == "Social Network Analysis":
    st.subheader("Social Network Analysis")

    # User interaction for selecting Entity Categories
    entity_categories = ['persons', 'organization', 'location']
    selected_category = st.selectbox("Select Entity Category", entity_categories)

    # Filter out data based on the selected Entity Category using str.contains
    filter_mask = trained_df['Entity_Categories'].apply(lambda x: selected_category in x)
    filtered_df = trained_df[filter_mask]

    st.write("### Filtered Data Preview")
    st.dataframe(filtered_df)

    if not filtered_df.empty:
        # Create a graph for social network analysis
        G = nx.Graph()

        # Populate the graph with nodes and edges based on filtered data
        for index, row in filtered_df.iterrows():
            topic = row['llama2_labelone']
            sentiment = row['llama3_sentiment']
            theme = row['cleaned_theme']

            # Add nodes
            G.add_node(topic, type='Topic')
            G.add_node(sentiment, type='Sentiment')
            G.add_node(theme, type='Theme')

            # Create edges based on relationships
            G.add_edge(topic, sentiment)
            G.add_edge(topic, theme)
            G.add_edge(sentiment, theme)

        # Create a mapping from node labels (e.g., strings) to integer indices
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}

        # Convert the edges to integer indices based on the node mapping
        edges = [(node_mapping[node1], node_mapping[node2]) for node1, node2 in G.edges]

        # Convert NetworkX graph edges to PyTorch tensor format
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        num_nodes = len(G.nodes)

        # Initialize identity matrix as features (can be modified)
        x = torch.eye(num_nodes, dtype=torch.float)  # Identity matrix as features (can be changed later)

        data = Data(x=x, edge_index=edge_index)

        # Simple GCN Model for Node Embeddings
        class GCN(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(GCN, self).__init__()
                self.conv1 = GCNConv(in_channels, 16)
                self.conv2 = GCNConv(16, out_channels)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return x

        # Instantiate the GCN model
        model = GCN(in_channels=num_nodes, out_channels=2)  # Output 2-dimensional embeddings

        # Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train the GCN model
        model.train()
        for epoch in range(200):  # Train for 200 epochs
            optimizer.zero_grad()
            out = model(data)
            loss = out.norm()  # Dummy loss function (use proper loss if needed)
            loss.backward()
            optimizer.step()

        # Get the learned embeddings from the last layer
        model.eval()
        with torch.no_grad():
            embeddings = model(data)

        # Use PCA to reduce dimensionality to 2D for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings.numpy())

        # Apply Spring Layout to optimize node positions and reduce overlap
        pos = nx.spring_layout(G, pos={node: (embeddings_2d[node_mapping[node], 0], embeddings_2d[node_mapping[node], 1]) for node in G.nodes()}, seed=42, k=0.15, iterations=300)

        # User interaction for selecting attributes for network structure
        st.write("### Select Attribute for Network Structure")
        structure_type = st.selectbox("Select Structure Type", ["Community", "Influence", "Link"])

        # Define attributes for different analyses
        community_attributes = ['Modularity', 'Degree', 'Clustering Coefficient']
        influence_attributes = ['Closeness Centrality', 'Betweenness Centrality', 'Eigenvector Centrality']
        link_attributes = ['Density', 'PageRank']

        G_selected = G.copy()  # Use the original graph

        if structure_type == "Community":
            selected_community_attribute = st.selectbox("Select Community Attribute", community_attributes)

            # Detect communities using the Louvain method
            partition = community_louvain.best_partition(G)

            # Assign weights based on selected community attribute
            if selected_community_attribute == "Modularity":
                modularity = community_louvain.modularity(partition, G)
                for node in G_selected.nodes():
                    G_selected.nodes[node]['weight'] = modularity  # This is a graph-wide metric

            elif selected_community_attribute == "Degree":
                degrees = dict(G.degree())
                for node in G_selected.nodes():
                    G_selected.nodes[node]['weight'] = degrees.get(node, 1)

            elif selected_community_attribute == "Clustering Coefficient":
                clustering = nx.clustering(G)
                for node in G_selected.nodes():
                    G_selected.nodes[node]['weight'] = clustering.get(node, 0)

        elif structure_type == "Influence":
            selected_influence_attribute = st.selectbox("Select Influence Attribute", influence_attributes)

            # Assign weights based on selected influence attribute
            if selected_influence_attribute == "Closeness Centrality":
                closeness = nx.closeness_centrality(G)
                for node in G_selected.nodes():
                    G_selected.nodes[node]['weight'] = closeness.get(node, 0)

            elif selected_influence_attribute == "Betweenness Centrality":
                betweenness = nx.betweenness_centrality(G)
                for node in G_selected.nodes():
                    G_selected.nodes[node]['weight'] = betweenness.get(node, 0)

            elif selected_influence_attribute == "Eigenvector Centrality":
                eigenvector = nx.eigenvector_centrality(G)
                for node in G_selected.nodes():
                    G_selected.nodes[node]['weight'] = eigenvector.get(node, 0)

        elif structure_type == "Link":
            selected_link_attribute = st.selectbox("Select Link Attribute", link_attributes)

            # Assign weights based on selected link attribute
            if selected_link_attribute == "Density":
                density = nx.density(G_selected)
                for node in G_selected.nodes():
                    G_selected.nodes[node]['weight'] = density  # Use density for each node

            elif selected_link_attribute == "PageRank":
                pagerank = nx.pagerank(G_selected)
                for node in G_selected.nodes():
                    G_selected.nodes[node]['weight'] = pagerank.get(node, 0)

        # User interaction for selecting the percentage of nodes to visualize
        percentage = st.slider("Select Percentage of Nodes to Visualize", 0, 100, 100)

        
        # Display the selected network graph only after the structure type is chosen
        if len(G_selected.nodes) > 0:
            st.write("### Selected Social Network Graph")
            # Determine the number of nodes to visualize
            num_nodes_to_visualize = max(1, int(len(G_selected.nodes()) * (percentage / 100)))
            selected_nodes = sorted(G_selected.nodes(), key=lambda n: G_selected.nodes[n]['weight'], reverse=True)[:num_nodes_to_visualize]

            G_filtered = G_selected.subgraph(selected_nodes)

            # plt.figure(figsize=(10, 8))
            # pos = nx.spring_layout(G_filtered)  # positions for all nodes
            # Use the PCA-reduced 2D embeddings for the positions of the nodes
            pos = {node: (embeddings_2d[node_mapping[node], 0], embeddings_2d[node_mapping[node], 1]) for node in G_filtered.nodes()}

            # Apply Spring Layout for additional refinement of node positions
            pos = nx.spring_layout(G_filtered, pos=pos, seed=42, k=0.15, iterations=300)  # This reduces overlap

            plt.figure(figsize=(10, 8))

            # Draw nodes with weight-based size and label
            node_sizes = [1000 * G_filtered.nodes[node]['weight'] for node in G_filtered.nodes()]  # Adjust size based on weight
            node_labels = {node: f"{node}" for node in G_filtered.nodes()}  # Labels without weights

            nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes, node_color='skyblue')
            nx.draw_networkx_edges(G_filtered, pos, alpha=0.5)
            nx.draw_networkx_labels(G_filtered, pos, labels=node_labels, font_size=8)
            plt.title(f"Social Network Visualization for Structure Type: {structure_type}")
            st.pyplot(plt)
            pyautogui.screenshot("screenshots/social_actors_screenshot.png")
            st.write("Screenshot of Social_actors_screenshot.png'.")

        else:
            st.warning("No nodes selected. Please select at least one category for the graph.")
    else:
        st.warning("The filtered DataFrame is empty. No relevant entities found.")
 

st.sidebar.markdown("### About this App")
st.sidebar.write("This app is designed for analyzing social network data and predictions, including sentiment analysis.")


# Conclusion of the app
st.sidebar.text("Developed by hsibitenda")

