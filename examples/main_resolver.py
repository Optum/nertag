"""Example script to resolve conflicting entities in a taxonomy"""

import sys

sys.path.append("..")

import pandas as pd
from nertag import utils, tool


path = "../data/taxonomy.csv" # path to taxonomy
df = pd.read_csv(path)

# --- Create reviser
reviser = tool.TaxonomyReviser(df)

reviser.preprocess(
    utils.preprocess_df,
    df,
    stemmer=utils.lemmatizer,
    filters=utils.stop_words,
    tokenizer=utils.tokenizer,
)

history = reviser.revise()
