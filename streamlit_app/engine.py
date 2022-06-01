from streamlit_multipage import MultiPage
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

import requests_cache
from datetime import timedelta
import pandas as pd
import json
from importlib.metadata import metadata

from crawler import scraper
from langprocessor import language_processor

requests_cache.install_cache('http_cache', expire_after=timedelta(days=1))


def input_page(st, **state):
    st.title("Crawler Engine Init-Params")

    url_ = state["url"] if "url" in state else None
    url = st.text_input("Domain name: ", value=url_)

    max_urls_ = state["max_urls"] if "max_urls" in state else 0
    max_urls = st.number_input("Max urls to crawl: ", value=max_urls_)
    clicked = st.checkbox("Run classifier")

    if clicked:
        classifiers_ = state["classifiers"] if "classifiers" in state else "[]"
        classifiers = st.text_input(
            "Enter custom classifier list: ", value=classifiers_)
        MultiPage.save({"max_urls": max_urls, "url": url,
                       "classifiers": eval(classifiers)})
    else:
        MultiPage.save({"max_urls": max_urls, "url": url, "classifiers": []})


@st.cache
@st.cache(suppress_st_warning=True)
def compute_page(st, **state):
    if "max_urls" not in state or "url" not in state:
        st.warning("Enter your data before crawler. Go to the Input Page")
        return

    with st.spinner("Please wait this might take a while..."):
        max_urls = state["max_urls"]
        url = state["url"]
        st.info("Crawling")
        s = scraper(url)
        s.begin(max_urls)
        forms_df = pd.DataFrame(s.all_forms.maps).dropna()

    st.success("Handing over exctracted data to language processor")
    st.info("Hang on, starting language processor")
    lp = language_processor()

    with st.spinner("Language processor running"):
        lp.process(s.knw_graph)

    classifiers = state["classifiers"]

    if classifiers:
        with st.spinner("custom classifier running"):
            st.info("Running custom classifier(might take a while)")

            c = lp.get_zero_shot_pipeline(labels=classifiers)
            lp.documents_df["custom_classified"] = lp.documents_df["original_docs"].apply(
                lp.classify_custom_labels)

    MultiPage.save({"knw_graph": s.knw_graph, "forms_df": forms_df, "internal_urls": s.internal_urls, "external_urls": s.external_urls, "dates": s.all_dates,
                   "tables": s.all_tables, "emails": s.emails, "phone": s.phone_numbers, "lp_df": lp.documents_df, "lp_ngrams": lp.n_grams, "lp_tfidf": lp.tfidf})

    st.success("Process complete. proceed to data explorer")


def url_data(st, **state):
    if "max_urls" not in state or "url" not in state:
        st.warning("Enter your data before crawler. Go to the Input Page")
        return

    with st.spinner("compiling website metadata"):
        url_home = state["url"]
        internal_urls = state["internal_urls"]
        external_urls = state["external_urls"]

        st.metric("No of Internal urls", value=len(list(internal_urls)))
        st.metric("No of External urls", value=len(list(external_urls)))

        st.write("Internal urls:")
        st.json(json.dumps(list(internal_urls)))
        st.write("External urls:")
        st.json(json.dumps(list(external_urls)))


def form_data(st, **state):
    if "max_urls" not in state or "url" not in state:
        st.warning("Enter your data before crawler. Go to the Input Page")
        return

    dates = state["dates"]
    emails = state["emails"]
    phone = state["phone"]

    df = state["forms_df"]
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gridOptions = gb.build()
    AgGrid(df, gridOptions=gridOptions)

    st.metric("No of dates mentions:", value=len(list(dates)))
    st.metric("No of emails ids:", value=len(list(emails)))
    st.metric("No of phone no:", value=len(list(phone)))

    st.write("dates mentions:")
    st.json(json.dumps(list(dates)))
    st.write("emails ids:")
    st.json(json.dumps(list(emails)))
    st.write("phone nos:")
    st.json(json.dumps(list(phone)))


def table_data(st, **state):
    if "max_urls" not in state or "url" not in state:
        st.warning("Enter your data before crawler. Go to the Input Page")
        return

    tables = state["tables"]

    st.metric("No of tables:", value=len(tables))
    for table in tables:
        st.write(table)


def lang_meta(st, **state):
    if "max_urls" not in state or "url" not in state:
        st.warning("Enter your data before crawler. Go to the Input Page")
        return

    lp_ngrams = state["lp_ngrams"]
    lp_tfidf = state["lp_tfidf"]

    st.subheader(
        "Common words weighted by their inter docs and intra docs freq:")
    st.write(lp_tfidf)
    st.subheader("Commonly occuring 2 words together(not weighted):")
    st.write(pd.DataFrame(
        lp_ngrams["esBigram_top_n"], columns=["n_gram", "repetition"]))
    st.subheader("Commonly occuring 3 words together(not weighted):")
    st.write(pd.DataFrame(
        lp_ngrams["esTrigram_top_n"], columns=["n_gram", "repetition"]))


def nlp_data(st, **state):

    df = state["lp_df"]
    st.subheader("General view:")
    gb = GridOptionsBuilder.from_dataframe(df[["original_docs", "origin_url"]])
    gb.configure_pagination()
    gridOptions = gb.build()
    AgGrid(df[["original_docs", "origin_url"]], gridOptions=gridOptions)

    st.subheader("NER view:")
    st.write("*(PER-person; ORG-organization; MISC-misc; LOC-location)")
    gb = GridOptionsBuilder.from_dataframe(
        df[["original_docs", "origin_url", "NER"]][~df["NER"].isna()])
    gb.configure_pagination()
    gridOptions = gb.build()
    AgGrid(df[["original_docs", "origin_url", "NER"]]
           [~df["NER"].isna()], gridOptions=gridOptions)

    st.subheader("Summary view:")
    gb = GridOptionsBuilder.from_dataframe(
        df[["original_docs", "origin_url", "summarized"]][~df["summarized"].isna()])
    gb.configure_pagination()
    gridOptions = gb.build()
    AgGrid(df[["original_docs", "origin_url", "summarized"]]
           [~df["summarized"].isna()], gridOptions=gridOptions)

    classifiers = state["classifiers"]
    if classifiers:
        st.subheader("Custom labeller:")
        gb = GridOptionsBuilder.from_dataframe(
            df[["original_docs", "origin_url", "custom_classified"]])
        gb.configure_pagination()
        gridOptions = gb.build()
        AgGrid(df[["original_docs", "origin_url", "custom_classified"]],
               gridOptions=gridOptions)


def header(st):
    st.write("Crawler WebApp")


app = MultiPage()
app.st = st

app.reset_button = "Del/Cache"
app.navbar_style = "VerticalButton"
app.header = header
app.start_button = "Go to the main page"
app.navbar_name = "Steps:"
app.next_page_button = "next step"
app.previous_page_button = "prev step"
# app.hide_menu = True
app.hide_navigation = True

app.add_app("Input Page", input_page)
app.add_app("Begin Crawl", compute_page)
app.add_app("Metadata Explorer", url_data)
app.add_app("Explore Misc Data", form_data)
app.add_app("Explore Table Data", table_data)
app.add_app("Language Metadata", lang_meta)
app.add_app("Explore NLP Features", nlp_data)


app.run()
