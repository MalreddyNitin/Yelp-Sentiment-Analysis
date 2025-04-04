import streamlit as st
import requests
import json
import re
import torch
import nltk
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoModelForSeq2SeqLM)

# -------------------- NLTK Setup --------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============== 1. SCRAPING (SERPAPI) ===============
def scrape_yelp_reviews(restaurant_name, city, api_key):
    """
    Scrape Yelp reviews for a given restaurant and city using SerpApi.
    Returns a list of review dictionaries (user_name, comment, rating, feedback).
    """
    # 1) Find the place_id by searching with 'find_desc' and 'find_loc'
    find_params = {
        "api_key": api_key,
        "engine": "yelp",
        "find_desc": restaurant_name,
        "find_loc": city
    }

    search = requests.get("https://serpapi.com/search", params = find_params)
    response = search.json()

    restaurant_place_id = next(item["place_ids"][0] for item in response["organic_results"] if item["title"]\
                              == restaurant_name)

    # 2) Paginate over reviews


    num = 49
    start = 0
    processed_reviews = []

    while True:
        review_params = {
            "api_key": api_key,
            "engine": "yelp_reviews",
            "place_id": restaurant_place_id,
            "start": start,
            "num": num
        }
        search_reviews = requests.get("https://serpapi.com/search", params=review_params)
        resp = search_reviews.json()

        reviews = resp.get("reviews", [])
        if not reviews:
            break

        for r in reviews:
            user_name = r["user"]["name"]
            comment_text = r["comment"]["text"]
            rating = r["rating"]
            feedback = r["feedback"]

            processed_reviews.append({
                "user_name": user_name,
                "comment": comment_text,
                "rating": rating,
                "feedback": feedback
            })

        if len(reviews) < num:
            break
        start += num

    return processed_reviews


# ============== 2. DATA CLEANING ===============
def clean_and_tokenize_reviews(review_data):
    """
    Given a list of reviews (dicts), clean and tokenize each 'comment'.
    Adds 'clean_comment' and 'tokens' fields to each review dict.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)           # remove HTML tags
        text = re.sub(r'[^a-z0-9\s]', '', text)     # keep only alphanumeric
        text = re.sub(r'\s+', ' ', text)            # remove multiple spaces
        return text.strip()

    def tokenize_and_lemmatize(text):
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    for review in review_data:
        comment = review.get("comment", "")
        cleaned = clean_text(comment)
        tokens = tokenize_and_lemmatize(cleaned)
        review["clean_comment"] = cleaned
        review["tokens"] = tokens

    return review_data


# ============== 3. SENTIMENT ANALYSIS (DistilBERT) ===============
@st.cache_resource
def load_distilbert_model():
    """
    Loads DistilBERT sentiment analysis model (SST-2).
    Cached to avoid re-loading on every run.
    """
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def sentiment_analysis(review_data):
    """
    For each review, classify sentiment (positive or negative) using DistilBERT.
    Adds 'distilbert_sentiment' field to each review dict.
    """
    tokenizer, model, device = load_distilbert_model()

    def classify_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        label_idx = torch.argmax(probs, dim=1).item()
        return "positive" if label_idx == 1 else "negative"

    for review in review_data:
        text = review.get("clean_comment", "")
        review["distilbert_sentiment"] = classify_sentiment(text)

    return review_data


# ============== 4. SUMMARIZATION (FLAN-T5) ===============
@st.cache_resource
def load_flan_t5_model():
    """
    Loads the FLAN-T5 model (base).
    Cached to avoid re-loading on every run.
    """
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def summarize_text(text, prompt="", max_length=150, min_length=40):
    """
    Summarize `text` using FLAN-T5 with `prompt`.
    """
    tokenizer, model, device = load_flan_t5_model()

    if not text.strip():
        return "No relevant reviews."

    input_text = f"summarize:\n{prompt}\n{text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        no_repeat_ngram_size=2,
        length_penalty=2.0,
        min_length=min_length,
        max_length=max_length,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()

def generate_summaries(review_data):
    """
    Creates two summaries:
    - For Customers (based on *all* text, but focusing on what's great or any warnings).
    - For Owner (based on *all* text, focusing on improvements, etc.).
    Returns a tuple: (customer_summary, owner_summary)
    """
    # Option A: Summarize positive & negative separately
    # We'll try combining them all for a "holistic" approach,
    # but you can adapt to separate positive_text and negative_text.
    all_comments = " ".join([r["comment"] for r in review_data])

    # 1) Customer Summary
    cust_prompt = (
        "Below are positive comments about the restaurant. "
            "Summarize them for a potential customer, focusing on popular dishes or great experiences:\n"
    )
    customer_summary = summarize_text(all_comments, cust_prompt)

    # 2) Owner Summary
    owner_prompt = (
        "Below are restaurant reviews from real customers, both positive and negative. "
    "Rewrite personal statements in third person. For example, if a review says 'I am from out of town,' "
    "rephrase it as 'Some visitors from out of town...'. "
    "Summarize from the restaurant owner's perspective, focusing on major compliments, common complaints, and suggestions to improve."
    )
    owner_summary = summarize_text(all_comments, owner_prompt)

    return customer_summary, owner_summary


# ============== 5. STREAMLIT APP ===============
def main():
    st.title("Restaurant Review Summaries")

    st.write("""
    Enter a restaurant name and city. We'll:
    1) Scrape Yelp reviews (via SerpApi)
    2) Clean and tokenize
    3) Perform sentiment analysis
    4) Summarize for both the **customer** and **restaurant owner** perspectives.
    """)

    api_key = st.text_input("Enter your SerpApi API Key", type="password")
    restaurant_name = st.text_input("Restaurant Name (e.g. 'The Globe')")
    city_name = st.text_input("City/Location (e.g. 'Madison')")

    if st.button("Analyze Reviews"):
        if not api_key or not restaurant_name or not city_name:
            st.error("Please provide an API key, restaurant name, and city.")
            return

        with st.spinner("Scraping Yelp Reviews..."):
            reviews_data = scrape_yelp_reviews(restaurant_name, city_name, api_key)

        if not reviews_data:
            st.error("No reviews found for that query. Try adjusting the name/city.")
            return

        st.success(f"Scraped {len(reviews_data)} reviews!")

        with st.spinner("Cleaning & Tokenizing..."):
            reviews_data = clean_and_tokenize_reviews(reviews_data)

        with st.spinner("Performing Sentiment Analysis..."):
            reviews_data = sentiment_analysis(reviews_data)

        with st.spinner("Generating Summaries..."):
            customer_summary, owner_summary = generate_summaries(reviews_data)

        st.subheader("Feedback for Customers")
        st.write(customer_summary)

        st.subheader("Feedback for the Restaurant Owner")
        st.write(owner_summary)



if __name__ == "__main__":
    main()
