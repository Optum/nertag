"""This module contains an interface for tagging named entities"""

import os
import tqdm
from tqdm.contrib.concurrent import process_map
from typing import Callable, List

from . import utils


class NER:
    def __init__(
        self,
        preprocessor: Callable[[str], str],
        baselabeler: Callable[[str], list],
        tagger: Callable[[list], list],
        verify: bool = True,
    ):
        self.preprocessor = preprocessor
        self.baselabeler = baselabeler
        self.tagger = tagger
        self.verify = verify

    def label(self, text: str):
        """Labels text using the following methodology:

        1. Labels all words to a background class
        2. Preprocess input text
        3. Labels processed text based on a taxonomy
        4. Aligns label offsets between (1 and 3)

        Args:
            text (str): string of text
            verify (bool, optional): flag to verify label correctness.

        Returns:
            list: labeled text
        """
        # --- Default labels
        labels = self.baselabeler(text)

        # --- Autolabeled ner tags
        words_and_offsets = self.preprocessor(text)
        tags = self.tagger(words_and_offsets)

        # --- Align labels with tags
        utils.align_labels(labels, tags)

        if self.verify:
            [
                utils.verify_tag(text, label["word"], label["start"], label["end"])
                for label in labels
            ]

        return labels

    def sequential_labeling(self, texts: list, position: int = 1, leave: bool = False):
        return [
            self.label(text)
            for text in tqdm.tqdm(texts, position=position, leave=leave)
        ]

    def parallel_labeling(
        self,
        texts: list,
        chunksize: int = 1,
        max_workers: int = os.cpu_count(),
        position: int = 1,
        leave: bool = False,
    ):
        return process_map(
            self.label,
            texts,
            max_workers=max_workers,
            chunksize=chunksize,
            position=position,
            leave=leave,
        )


# --- Interfaces
class Preprocessor:
    """Class to define text preprocessing logic"""

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, text: str):
        return self.func(text, *self.args, **self.kwargs)


class BaseLabeler:
    """Class to define initial labeling behavior"""

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, text: str):
        return self.func(text, *self.args, **self.kwargs)


class Tagger:
    """Class to define named-entity tagging logic"""

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dct: dict):
        return self.func(dct, *self.args, **self.kwargs)
