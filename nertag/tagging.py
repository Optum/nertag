"""This module contains functions for tagging named entities"""

from . import utils


def ner_tagging(words_and_offsets: dict, dct: dict, tokenizer=utils.tokenizer):
    """Returns tagged terms with corresponding entity, word location, and start / end index.

    The algorithm iterates over a set of n_grams and indices (words_and_offsets), keeping
        track of terms that have been tagged. The algorithm is greedy and only tags entities
        from a specific index once, thus, n_grams should be ordered from largest to smallest.

    Example:
    - Inputs (words_and_offsets):
        {
            "words": [
                ["account lock account use", "lock account use unlock"],
                ["account lock account", "lock account use", "account use unlock"],
                ["account lock", "lock account", "account use", "use unlock"],
                ["account", "lock", "account", "use", "unlock"],
            ],
            "offsets": [
                ["1 3 5 8", "3 5 8 10"],
                ["1 3 5", "3 5 8", "5 8 10"],
                ["1 3", "3 5", "5 8", "8 10"],
                ["1", "3", "5", "8", "10"],
            ],
        }


    - Outputs:
    [
        {'entity': 'B-common1k', 'word': 'order', 'index': 3},
        {'entity': 'B-FSA:Flexible Spending Account', 'word': 'fsa', 'index': 5}
    ]

    Args:
        words_and_offsets (dict): dictionary containing word n_grams and offsets (see preprocess for reference)
        dct (dict): dictionary referencing the taxonomy (key: example, value: entity)
        tokenizer (func, optional): function to tokenize text. Defaults to utils.tokenizer.

    Returns:
        list[dict]: the tagged terms as a list of dictionaries
    """

    tags = []
    bag = set()

    n_grams = words_and_offsets["words"]
    offsets = words_and_offsets["offsets"]

    # --- Loop over n_grams
    for i, (n_gram, offset) in enumerate(zip(n_grams, offsets)):

        # --- Loop over each n_gram
        for words, indices in zip(n_gram, offset):
            words = tokenizer(words)
            indices = tokenizer(indices)

            if not any([i for i in indices if i in bag]):
                # --- Find corresponding entity
                entity = get_match(words, dct)

                if entity:
                    # --- Store match
                    tags += [
                        utils.tag_beginning(entity, word, int(index))
                        if i == 0
                        else utils.tag_inside(entity, word, int(index))
                        for i, (word, index) in enumerate(zip(words, indices))
                    ]
                    bag.update(indices)

    return tags


def get_match(words: list, dct: dict) -> str:
    """Returns corresponding entity if possible"""

    words = frozenset(words)
    if words in dct:
        return dct[words]
