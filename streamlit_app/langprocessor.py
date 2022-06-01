import re
import string
import unicodedata
import nltk
import contractions
import inflect
from nltk import collections
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import pipeline
import time
from colorama import init, Fore, Back, Style  # use init for term
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from collections import ChainMap, OrderedDict, namedtuple
import json
import streamlit as st


class language_processor:
    def __init__(self):
        self.docs = []
        self.doc_clean = []

    def replace_contractions(self, text):
        return contractions.fix(text)

    def remove_URL(self, sample):
        return re.sub(r"http\S+", "", sample)

    def remove_non_ascii(self, words):
        new_words = []
        for word in words:
            new_word = (
                unicodedata.normalize("NFKD", word)
                .encode("ascii", "ignore")
                .decode("utf-8", "ignore")
            )
            new_words.append(new_word)
        return new_words

    def to_lowercase(self, words):
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self, words):
        new_words = []
        for word in words:
            new_word = re.sub(r"[^\w\s]", "", word)
            if new_word != "":
                new_words.append(new_word)
        return new_words

    def replace_numbers(self, words):
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self, words):
        new_words = []
        for word in words:
            if word not in stopwords.words("english"):
                new_words.append(word)
        return new_words

    def stem_words(self, words):
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self, words):
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos="v")
            lemmas.append(lemma)
        return lemmas

    def normalize(self, words):
        words = self.remove_non_ascii(words)
        words = self.to_lowercase(words)
        words = self.remove_punctuation(words)
        words = self.replace_numbers(words)
        words = self.remove_stopwords(words)
        return words

    def preprocess(self, sample):
        sample = self.remove_URL(sample)
        sample = self.replace_contractions(sample)
        words = nltk.word_tokenize(sample)
        words = self.normalize(words)
        res = " ".join(words)
        return res

    def create_doc_list(self, knw_graph):
        u = []
        for _, maps in enumerate(knw_graph.maps):
            if maps:
                if maps["para_list"]:
                    pl = maps["para_list"]

                    for _, p in enumerate(pl):
                        u.append(maps["url"])
                        self.docs.append(p)
        return self.docs, u

    def recursive_drop_similar(
        self, documents_df, pairwise_similarities, threshold=0.80
    ):
        id = set()
        tic = time.perf_counter()
        for i in range(len(pairwise_similarities)):
            similar_ix = np.argsort(pairwise_similarities[i])[::-1]
            for ix in similar_ix:
                if ix == i:
                    continue
                elif pairwise_similarities[i][ix] > 0.85:
                    id.add(ix)
        toc = time.perf_counter()
        print(f"~ took {toc - tic:0.4f} seconds to drop redunduncies")

        return (
            documents_df.drop(list(id), axis=0)
            .reset_index()
            .rename(columns={"index": "original_index"})
        )

    def remove_text_redundancy(self, knw_graph):
        print(Fore.GREEN+"[*] truncating corpus using similarity metric")
        st.info("Truncating corpus using similarity metric")
        self.docs, u = self.create_doc_list(knw_graph)
        tic = time.perf_counter()

        for _, text in enumerate(self.docs):
            self.doc_clean.append(self.preprocess(text))

        documents_df_org = pd.DataFrame(self.doc_clean, columns=["documents"])
        documents_df_org["original_docs"] = self.docs
        documents_df_org["origin_url"] = u

        sbert_model = SentenceTransformer("bert-base-nli-mean-tokens")
        document_embeddings = sbert_model.encode(documents_df_org["documents"])
        pairwise_similarities = cosine_similarity(document_embeddings)

        documents_df = self.recursive_drop_similar(
            documents_df_org, pairwise_similarities
        )
        toc = time.perf_counter()
        print(f"~ took {toc - tic:0.4f} seconds to clean")

        return documents_df

    @staticmethod
    def most_similar(documents_df, doc_id, similarity_matrix):
        print(f'Document: {documents_df.iloc[doc_id]["documents"]}')
        print("\n")
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
        for ix in similar_ix:
            if ix == doc_id:
                continue
            print("\n")
            print(f'Document: {documents_df.iloc[ix]["documents"]}')
            print(f"{similarity_matrix[doc_id][ix]}")

    def n_gram_generator(self, doc, custom_g=False, top_n=10):
        print(Fore.GREEN+"[*] creating n-grams")
        st.info("Creating n-grams")
        joined_doc = " ".join(doc)
        tokenized = joined_doc.split()
        esBigram_top_n = []
        esTrigram_top_n = []
        esCustomgramFreq = []

        esBigrams = ngrams(tokenized, 2)
        esBigramFreq = collections.Counter(esBigrams)
        esBigram_top_n = esBigramFreq.most_common(top_n)

        esTrigrams = ngrams(tokenized, 3)
        esTrigramFreq = collections.Counter(esTrigrams)
        esTrigram_top_n = esTrigramFreq.most_common(top_n)

        if custom_g:
            esCustomgrams = ngrams(tokenized, custom_g)
            esCustomgramFreq = collections.Counter(esCustomgrams)
            esCustomgram_top_n = esCustomgramFreq.most_common(top_n)
            return OrderedDict(
                esBigram_top_n=esBigram_top_n,
                esTrigram_top_n=esTrigram_top_n,
                esCustomgram_top_n=esCustomgram_top_n,
            )
        return OrderedDict(
            esBigram_top_n=esBigram_top_n, esTrigram_top_n=esTrigram_top_n
        )

    def get_ner_pipeline(self):
        print(Fore.GREEN+"[*] loading ner pipeline")
        tokenizer = AutoTokenizer.from_pretrained(
            "dslim/bert-base-NER-uncased")
        model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-base-NER-uncased"
        )
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        return nlp

    def get_zero_shot_pipeline(self, labels):
        self.labels = labels
        print(Fore.GREEN+"[*] loading zero shot classifier pipeline ")
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli"
        )
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.classifier = pipeline(
            "zero-shot-classification", model=nli_model, tokenizer=tokenizer
        )
        return self.classifier

    def classify_custom_labels(self, text):
        return self.classifier(text, self.labels, multi_label=False)

    def get_summarizer_pipeline(self):
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    def apply_ner(self, x):
        y = self.nlp(x)
        if y:
            return y
        else:
            return np.nan

    def create_summary(self, x):
        if len(x) > 400:
            return self.summary_pipeline(x,  min_length=20, max_length=60)[0]["summary_text"]
        else:
            return np.nan

    def get_tfidf(self):
        print(Fore.GREEN+"[*] creating tfidf matrix")
        st.info("Creating tfidf matrix")

        test_list = []
        for _, i in enumerate(self.documents_df["documents"]):
            test_list.append(i)

        tfIdfTransformer = TfidfTransformer(use_idf=True)
        countVectorizer = CountVectorizer()

        wordCount = countVectorizer.fit_transform(test_list)
        newTfIdf = tfIdfTransformer.fit_transform(wordCount)

        self.tfidf = pd.DataFrame(newTfIdf[0].T.todense(
        ), index=countVectorizer.get_feature_names_out(), columns=["TF-IDF"])
        self.tfidf = self.tfidf.sort_values('TF-IDF', ascending=False)

    def process(self, knw_graph):

        self.documents_df = self.remove_text_redundancy(knw_graph)
        self.n_grams = self.n_gram_generator(self.doc_clean)
        self.nlp = self.get_ner_pipeline()
        print(Fore.GREEN+"[*] generating NER tokens ")
        st.info("Generating NER tokens ")
        self.documents_df["NER"] = self.documents_df["original_docs"].apply(
            self.apply_ner)
        print(Fore.GREEN+"[*] tokenizing")
        st.info("Tokenizing")
        self.documents_df["sent_tokenized"] = self.documents_df["original_docs"].apply(
            sent_tokenize
        )
        self.get_tfidf()
        self.summary_pipeline = self.get_summarizer_pipeline()
        print(Fore.GREEN+"[*] generating summary ")
        st.info("Generating summary for large text data")
        st.warning("this will take a while!")
        tic = time.perf_counter()
        self.documents_df["summarized"] = self.documents_df["original_docs"].apply(
            self.create_summary)
        toc = time.perf_counter()
        print(f"~ took {toc - tic:0.4f} seconds to generate summary")
