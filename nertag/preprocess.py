"""This module contains functions to preprocess text"""

from . import utils


def normalize(text: str, stemmer=utils.lemmatizer, tokenizer=utils.tokenizer):
    """Stems text and removes punctuation

    Args:
        text (str): string of text
        stemmer (func): function to stem text. Defaults to utils.lemmatizer.
        tokenizer (func, optional): function to tokenize text. Defaults to utils.tokenizer.

    Returns:
        str: normalized text
    """

    text = utils.remove_punct(text)
    text = utils.stem_text(text, stemmer, tokenizer)

    return text


def get_words(text: str, stop_words=utils.stop_words, tokenizer=utils.tokenizer):
    """Get words

    Args:
        text (str): string of text
        stop_words (list): list of words or terms to ignore. Defaults to utils.stop_words.
        tokenizer (func, optional): function to tokenize text. Defaults to utils.tokenizer.

    Returns:
        list: a collection of words
    """

    words = tokenizer(text)
    words = utils.filter_words(words, stop_words)
    words = [word for word in words]

    return words


def get_offsets(text: str, stop_words=utils.stop_words, tokenizer=utils.tokenizer):
    """Get word offsets

    Args:
        text (str): string of text
        stop_words (list): list of words or terms to ignore. Defaults to utils.stop_words.
        tokenizer (func, optional): function to tokenize text. Defaults to utils.tokenizer.

    Returns:
        list: a collection of word indices
    """

    words = tokenizer(text)
    words = utils.filter_words_offset(words, stop_words)
    words = [str(word) for word in words]

    return words


def preprocess(
    text,
    stemmer=utils.lemmatizer,
    stop_words=utils.stop_words,
    tokenizer=utils.tokenizer,
    start=4,
    stop=0,
    step=-1,
):
    """Preprocesses and converts text into varying length n_grams with corresponding word offsets

    Args:
        text (str): string of text
        stemmer (func): function to stem text. Defaults to utils.lemmatizer.
        stop_words (list): list of words or terms to ignore
        tokenizer (func, optional): function to tokenize text. Defaults to utils.tokenizer.
        start (int, optional): initial n for n_grams. Defaults to 4.
        stop (int, optional): final n for n_grams. Defaults to 0.
        step (int, optional): step size for next n_gram. Defaults to -1.

    Returns:
        dict: dictionary containing a list of n_grams and word offsets
    """

    # Time-complexity
    # O(num_grams * num_words_per_gram)
    # ~ O(k * n)

    text = normalize(text, stemmer)

    words = get_words(text, stop_words, tokenizer)
    offsets = get_offsets(text, stop_words, tokenizer)

    words = utils.get_ngrams(words, start, stop, step)
    offsets = utils.get_ngrams(offsets, start, stop, step)

    return {"words": words, "offsets": offsets}
