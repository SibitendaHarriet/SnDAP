# SnDAP
**Analyzing social network data about development social concerns**

---

## Overview
**SnDAP** (Social Network Data Analysis and Prediction) is a web-based application for analyzing and predicting themes, topics, and sentiment trends related to development and social concerns. It enables users to explore pre-trained datasets, visualize insights, and upload their own data for predictions.

---

## Features
### 🔍 **Data Analysis**
- **Topics Analysis**: Generate visualizations like Word Clouds, Histograms, and Bar Charts to explore key topics.
- **Themes Analysis**: Classify content into themes like poverty, education, and health. Visualize data distributions using Word Clouds and Bar Charts.
- **Sentiment Analysis**: Analyze Positive, Neutral, or Negative sentiment trends across datasets.

### 📈 **Temporal Analysis**
- **Analyze Themes and Topics Over Time**:
  - Visualize the evolution of themes and topics over **daily**, **weekly**, **monthly**, or **yearly** timeframes.
  - Use dynamic filtering options to refine results by frequency ranges and sentiment categories.
  - Gain insights into trends through time-series plots, highlighting the sentiment-specific progression of key topics or themes.
  
- **Forecast Trends with Prophet**:
  - Use forecasting models to predict future trends for selected themes or topics.
  - Visualize forecasts alongside historical data, including confidence intervals, for deeper insights.

### 🌐 **Social Network Analysis**
- **Explore Entity Relationships**:
  - Analyze relationships among entities such as persons, organizations, and locations in social network datasets.
  - Filter datasets based on specific entity categories.

- **Build Social Networks**:
  - Create interactive graph visualizations to depict connections between **Topics**, **Themes**, and **Sentiments**.
  - Nodes and edges are generated dynamically from relationships in the data.

- **Node Embedding and Graph Analysis**:
  - Utilize advanced techniques like **Graph Convolutional Networks (GCNs)** and **PCA** for embedding nodes into low-dimensional space.
  - Visualize nodes based on **Community**, **Influence**, or **Link** attributes, with metrics like:
    - **Modularity**, **Clustering Coefficients**, and **Degree** (for community analysis).
    - **Closeness Centrality**, **Betweenness Centrality**, and **Eigenvector Centrality** (for influence analysis).
    - **PageRank** and **Network Density** (for link analysis).

- **Dynamic Graph Filtering**:
  - Filter graphs by selecting a percentage of nodes to visualize, prioritizing high-weighted entities for clarity.
  - Generate exportable graph screenshots for reporting.

### 📊 **Custom Predictions**
- Upload your dataset in CSV format for prediction and analysis of themes, topics, and sentiment trends.
- Download analyzed results or visualizations for future use.

### 🎨 **Interactive Visualizations**
- Create and explore dynamic visualizations powered by **Matplotlib**, **Seaborn**, and **Plotly**.

---

## Installation
Follow these steps to set up SnDAP on your local system.

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SibitendaHarriet/sndap.git
   cd sndap
# Documentation for Social Concerns Africa

Welcome to the **Social Concerns Africa** documentation site. This guide provides resources and step-by-step instructions for collecting, preparing, and analyzing social media data to extract valuable insights.

---

## Overview

This documentation covers the following areas:
- **Data Collection**: How to collect data from platforms like Twitter and YouTube.
- **Data Preparation**: Methods to clean and preprocess collected data.
- **Extraction of Social Concerns**: Techniques for topic modeling, document similarity, and sentiment analysis.
- **Analysis of Social Concerns**: Temporal and social network analysis to uncover trends and patterns.
- **Visualization of Concerns**: Building dashboards to present insights effectively.

---

## Quick Links

- [Twitter Data Collection](1_Data_collection/twitter-data-collection.md)
- [YouTube Data Collection](1_Data_collection/youtube_data_collection.md)
- [Data Preparation](2_Data_preparation/data_preparation.md)
- [Topic Modeling](3_Extraction_of_social_concerns/Topic_modeling.md)
- [Sentiment Analysis](3_Extraction_of_social_concerns/Sentiment_analysis.md)
- [Temporal Analysis](4_Analysis_of_concerns/Temporal_analysis.md)
- [Social Network Analysis](4_Analysis_of_concerns/Social_network_analysis.md)
- [Dashboard Visualization](5_Visualisation_of_concerns/dashboard.md)

---

Explore the sidebar or use the links above to get started!

