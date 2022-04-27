"""Example script to autolabel named-entities"""

import sys

sys.path.append("..")

import tqdm
import string
import pandas as pd
from nertag import ner, preprocess, utils, tagging


def file_generator(path, chunk_size=1024):
    with open(path, "r") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data


def batch(gen, batch_size=64):
    data = []
    for _ in range(batch_size):
        try:
            dat = next(gen)
            data.append(dat)
        except StopIteration:
            break

    return data


def reformat(gen):
    texts = []
    for text in gen:
        text = text.replace(".", ". ")
        text = " ".join(text.split())
        texts.append(text)

    return " ".join(texts)


def num_batches(gen):
    return sum([1 for _ in gen if batch(gen)])


def skip_batch(gen, n_batches, stop):
    for _ in range(stop):
        batch(gen)


def custom_filter(text: str, filters: set = set(string.punctuation)):
    return " ".join([word for word in text.split() if word not in filters])


if __name__ == "__main__":

    # --- Taxonomy
    path = "../data/taxonomy.csv"  # path to taxonomy
    df = utils.preprocess_df(
        pd.read_csv(path),
        stemmer=utils.lemmatizer,
        filters=utils.stop_words,
        tokenizer=utils.tokenizer,
    )
    dct = utils.setup_dict(df)

    # --- Data
    path = "../data/data.txt"  # path to unstructured text
    gen = file_generator(path)
    n_batches = num_batches(gen)

    # --- Pipeline
    preprocessor = ner.Preprocessor(
        preprocess.preprocess,
        stemmer=utils.lemmatizer,
        stop_words=utils.stop_words,
        start=4,
        stop=0,
        step=-1,
    )
    baselabeler = ner.BaseLabeler(utils.base_label, utils.tokenizer)
    tagger = ner.Tagger(tagging.ner_tagging, dct)
    pipeline = ner.NER(preprocessor, baselabeler, tagger)

    # --- Seek
    gen = file_generator(path)
    start = 0
    skip_batch(gen, n_batches, start)

    # --- Label
    for i in tqdm.tqdm(range(start, n_batches), position=0):
        texts = batch(gen)
        texts = [custom_filter(text) for text in texts]

        # NOTE: Better for large datasets, however, for very large taxonomies 
        # (high RAM usage), sequential_labeling will be better
        # results = pipeline.parallel_labeling(texts, chunksize=16)

        results = pipeline.sequential_labeling(texts)
        
        # NOTE: This should really be done as a background task
        utils.to_dataframe(results).to_csv(f"../data/label_{i}.csv", index=False)
