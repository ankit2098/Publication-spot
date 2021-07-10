import pandas as pd
import streamlit as st
import nltk
nltk.download("wordnet")
import streamlit_analytics
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import texthero as hero
from nltk.stem import WordNetLemmatizer
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import ssl
import time
from random import randint
from scraper_api import ScraperAPIClient
import concurrent.futures

client = ScraperAPIClient('13e21e2becfc6f6f0880ca408bd91d47') #API Key

###define global variables
global extracted_titles
global extracted_abstracts
global extracted_links
global urls
extracted_titles = []
extracted_abstracts = []
extracted_links = []
urls = []

def textprocessing(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text)
    text = pd.Series(text)
    text = hero.clean(text)
    cleaned_text = hero.remove_stop_words(text)
    return cleaned_text[0]

def extractkeywords(text):
    #vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X2 = vectorizer.fit_transform([text])
    features = vectorizer.get_feature_names()
    scores = (X2.toarray())
    #Getting top ranking features
    sums = X2.sum(axis=0)
    data = []
    for col, term in enumerate(features):
        data.append((term, sums[0, col]))
    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    keywords = ranking.sort_values('rank', ascending=False)
    return keywords

def extractgooglescholararticle(keywords):
    #time.sleep(randint(5,10))
    query = " ".join(word for word in keywords)
    query = query.split()
    #query = list(dict.fromkeys(query))   #remove duplicates from the keywords
    string = []
    for i in range(0, len(query)):
        if i < len(query) - 1:
            string.append(query[i] + "+")
        else:
            string.append(query[i])
    query = "".join(word for word in string)
    i = 0
    while i<20:
        g_url = "https://scholar.google.com/scholar?start=" + str(i) + "&q=" + query + "&hl=en&as_sdt=0,5"
        urls.append(g_url)
        i+=10
    return

def extractsemnaticscholars(keywords):
    query = " ".join(word for word in keywords)
    query = query.split()
    query = list(dict.fromkeys(query))  # remove duplicates from the keywords
    string = []
    for i in range(0, len(query)):
        if i < len(query) - 1:
            string.append(query[i] + "%20")
        else:
            string.append(query[i])
    query = "".join(word for word in string)
    article_titles = []
    article_abstracts = []
    article_links = []
    i = 1
    index = 0
    while i < 2:
        url = "https://www.semanticscholar.org/search?q=" + query + "&sort=relevance&page=" + str(i)
        page = client.get(url, render="True")
        soup = BeautifulSoup(page.content, 'html.parser')
        titles = soup.find_all('div', class_='cl-paper-title')
        abstracts = soup.find_all("div", class_='cl-paper-abstract')
        links = soup.find_all("div", class_='cl-paper-row serp-papers__paper-row paper-row-normal')
        for title, abstract, link in zip(titles, abstracts, links):
            if title != None and abstract != None and link.a != None:
                article_titles.append(title.text)
                abstract = re.sub("TLDR", "", abstract.text)
                abstract = re.sub("Expand", "", abstract)
                article_abstracts.append(abstract)
                article_links.append("https://www.semanticscholar.org/" + link.a['href'])
                index += 1
        i += 1
    return article_titles, article_abstracts, article_links

def relatedgooglescholararticle(title):
    title = re.sub(r'[^\w\s]', '', title)
    keywords = title.split()
    words = []
    for i in range(0, len(keywords)):
        if i < len(keywords) - 1:
            words.append(keywords[i] + "+")
        else:
            words.append(keywords[i])
    query = "".join(word for word in words)
    url = "https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=" + query + "&btnG="
    page = client.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    id = soup.find("div", class_="gs_r gs_or gs_scl")
    pid = id["data-cid"]
    i = 0
    while i<20:
        rg_url = "https://scholar.google.com/scholar?start=" + str(i) + "&q=related:" + str(pid) + ":scholar.google.com/&hl=en&as_sdt=0,5&scioq=" + query
        urls.append(rg_url)
        i+=10
    return

def extractdata(url):
    page = client.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    meta_titles = soup.select('.gs_rt')
    meta_abstracts = soup.select('.gs_rs')
    meta_links = soup.select('.gs_rt')
    # for result in soup.select('.gs_ri'):
    for title, abstract, link in zip(meta_titles, meta_abstracts, meta_links):
        if title != None and abstract != None and link.a != None:
            extracted_titles.append(title.text)
            extracted_abstracts.append(abstract.text)
            extracted_links.append(link.a['href'])
    return

def makecorpus(titles, abstracts):
    corpus = []
    for i in range(0, len(titles)):
        title = titles[i]
        abstract = abstracts[i]
        summary = title + ". " + abstract
        corpus.append(textprocessing(summary))
    return corpus

def score(y, X, titles, abstracts, links):
    index = 1
    for i in range(len(titles)):
        similarity = cosine_similarity(y[0].reshape(1,-1), X[i].reshape(1,-1))
        if similarity>0.1:
            st.subheader(titles[i])
            st.write(abstracts[i])
            st.write(links[i])
            #st.text(similarity)
            index+=1
    return 0

def tfidf(query, corpus):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(corpus).toarray()
    y = tfidf.transform([query]).toarray()
    #print(X.shape)
    return X, y

#-----Main function starts-----
streamlit_analytics.start_tracking()  #analytics
st.markdown('# Publication Spot')
st.markdown("** Do literature search based on the paper you find useful, not using keywords! **")
st.markdown("** Just input the title and abstract of the article, and get similar articles **")
st.write("Enter Title ")
title = st.text_input("")
st.write("Enter Abstract")
abstract = st.text_area("", height= 300)

#hiding the side bar
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

if st.button(label="Get recommendations"):
    if not title:
        st.markdown("Please enter a query")
    else:
        summary = title +"." +abstract
        cleaned_summary = textprocessing(summary)
        keywords_summary = extractkeywords(cleaned_summary)
        keywords_summary = keywords_summary.to_numpy()
        keywords = []
        i =0
        while i < 5:
            keywords.append((keywords_summary[i][0]))
            i+=1
        # Google scholar
        extractgooglescholararticle(keywords)
        # related gs articles
        relatedgooglescholararticle(title)
        #Multithreading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(extractdata, urls)
        # tfidf implementation
        corpus = makecorpus(extracted_titles, extracted_abstracts)
        X, y = tfidf(cleaned_summary,corpus)
        score(y, X, extracted_titles, extracted_abstracts, extracted_links)
streamlit_analytics.stop_tracking()
