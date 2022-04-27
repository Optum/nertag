"""This module contains a variety of utility functions"""

import string
import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from typing import List


# --- NLTK


stop_words = stopwords.words("english")
tokenizer = str.split
stemmer = PorterStemmer().stem
lemmatizer = WordNetLemmatizer().lemmatize
punct_map = str.maketrans("", "", string.punctuation)


# --- IOB2 Tag schema (inside-outside-beginning tagging)


class Tag:
    beginning = "B"
    inside = "I"
    background = "O"


# --- Text processing


def remove_punct(text: str, mapping: set = punct_map) -> str:
    return text.translate(mapping)


def stem_text(text: str, stemmer, tokenizer) -> str:
    return " ".join([stemmer(t) for t in tokenizer(text)])


def filter_words(text: list, filters: list) -> list:
    return [t for t in text if t not in filters]


def filter_words_offset(text: list, filters: list) -> list:
    return [i for i, t in enumerate(text) if t not in filters]


def join_grams(gram: list) -> list:
    return [" ".join(terms) for terms in gram]


def get_ngrams(text: list, start: int, stop: int, step: int) -> List[list]:
    grams = [ngrams(text, n) for n in range(start, stop, step)]
    terms = [join_grams(gram) for gram in grams]
    return terms


# --- Taxonomy loading


def preprocess_df(df: pd.DataFrame, stemmer, filters: list, tokenizer) -> pd.DataFrame:
    df = stem_df(df, stemmer, tokenizer)
    df = set_df(df, tokenizer)

    return df


def stem_df(df: pd.DataFrame, stemmer, tokenizer) -> pd.DataFrame:
    columns = [
        df[column].dropna().apply(lambda words: stem_text(words, stemmer, tokenizer))
        for column in df.columns
    ]

    return pd.concat(columns, axis=1)


def filter_df(df: pd.DataFrame, filters: list, tokenizer) -> pd.DataFrame:
    columns = [
        df[column]
        .dropna()
        .apply(lambda words: " ".join(filter_words(tokenizer(words), filters)))
        for column in df.columns
    ]

    return pd.concat(columns, axis=1)


def set_df(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    columns = [
        df[column].dropna().apply(lambda words: frozenset(tokenizer(words)))
        for column in df.columns
    ]

    return pd.concat(columns, axis=1)


def unique_df(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    columns = [df[column].dropna().dropna().drop_duplicates() for column in df.columns]

    return pd.concat(columns, axis=1)


def to_dataframe(arr: list, axis=0):
    return pd.concat([pd.DataFrame(a) for a in arr], axis=axis)


def setup_dict(df: pd.DataFrame) -> dict:
    return {example: column for column in df.columns for example in df[column].dropna()}


# --- Taxonomy saving


def drop_tags(labels: list, tag: str) -> list:
    return [
        {k: v for k, v in label.items()}
        for label in labels
        if label["entity"][0] != tag
    ]


def merge_tags(labels: list) -> list:
    """Merge begin and inside tags"""

    for i, label in enumerate(labels):
        if label["entity"][0] == Tag.beginning:
            label["entity"] = label["entity"][2:]
            begin = i

        elif label["entity"][0] == Tag.inside:
            labels[begin]["word"] += " " + label["word"]


def entity_tags(labels: list) -> list:
    labels = drop_tags(labels, Tag.background)
    merge_tags(labels)
    labels = drop_tags(labels, Tag.inside)

    return labels


def to_taxonomy(labels: List[list]) -> pd.DataFrame:
    """Transforms NER labels to a taxonomy dataframe"""

    results = []
    for tags in labels:
        tags = entity_tags(tags)
        results += tags

    taxonomy = {}
    for label in results:
        entity = label["entity"]
        word = label["word"]
        if entity not in taxonomy:
            taxonomy[entity] = []
        taxonomy[entity] += [word]

    taxonomy = {k: sorted(list(set(v))) for k, v in taxonomy.items()}

    return to_dataframe(({entity: taxonomy[entity]} for entity in taxonomy), axis=1)


# --- Labeling


def tag_word(entity, word, index) -> dict:
    return {"entity": entity, "word": word, "index": index}


def tag_beginning(entity: str, word: str, index: int):
    return tag_word(f"{Tag.beginning}-{entity}", word, index)


def tag_inside(entity: str, word: str, index: int):
    return tag_word(f"{Tag.inside}-{entity}", word, index)


def align_labels(labels: list, tags: list) -> None:
    for tag in tags:
        index = tag["index"]
        entity = tag["entity"]
        labels[index]["entity"] = entity

        reverse_tag(labels, tag)


def reverse_tag(labels: list, tag: dict) -> None:
    index = tag["index"]
    entity = tag["entity"]

    if entity[0] == Tag.inside:
        for i in range(index - 1, 0, -1):
            label = labels[i]
            if label["entity"][0] == Tag.background:
                label["entity"] = entity
            else:
                break


def verify_tag(text: str, word: str, start: int, end: int) -> bool:
    """Verify NER tagged word

    Args:
        text (str): string of text
        word (str): ner labeled word
        start (int): start index of word
        end (int): end index of word

    Returns:
        bool: True if correct False otherwise
    """

    assert text[start:end] == word


def base_label(text: str, tokenizer) -> list:
    """Defines the initial labeling behavior

    Args:
        text (str): string of text
        tokenizer (func, optional): function to tokenize text.

    Returns:
        list: initial labeles for each word
    """

    labels = []
    begin = 0

    for i, t in enumerate(tokenizer(text)):
        label = {
            "entity": Tag.background,
            "word": t,
            "index": i,
            "start": begin,
            "end": begin + len(t),
        }
        labels.append(label)
        begin += len(t) + 1

    return labels
