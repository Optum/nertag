"""This module contains a tool for resolving conflicts within the taxonomy"""

import sys
import json
import pandas as pd
import logging
from . import utils


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def json_dump(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


class TaxonomyReviser:
    UNK = "[UNK]"

    def __init__(self, df_raw: pd.DataFrame, df_proc: pd.DataFrame = None):
        self.df_raw = df_raw.copy()
        self.df_proc = df_proc if df_proc is None else df_proc.copy()

        self.set2ent = {}  # store mapping between word set and entity
        self.set2wrd = {}  # store mapping between word set and original word

    def preprocess(self, func, *args, **kwargs):
        self.df_proc = func(*args, **kwargs)

        assert all(self.df_raw.columns == self.df_proc.columns)
        assert all(
            [
                len(self.df_raw[column]) == len(self.df_proc[column])
                for column in self.df_raw.columns
            ]
        )

    def set_mappings(self):
        self.set2ent = {}
        self.set2wrd = {}

        for column in self.df_raw.columns:
            raws = self.df_raw[column].dropna()
            procs = self.df_proc[column].dropna()

            for raw, proc in zip(raws, procs):
                proc = frozenset(proc)

                if proc not in self.set2wrd:
                    self.set2wrd[proc] = []

                if proc not in self.set2ent:
                    self.set2ent[proc] = []

                self.set2wrd[proc] += [raw]
                self.set2ent[proc] += [column]

        self.set2wrd = {k: list(set(v)) for k, v in self.set2wrd.items()}
        self.set2ent = {k: list(set(v)) for k, v in self.set2ent.items()}

    def get_mappings(self):
        return {"set2wrd": self.set2wrd, "set2lbl": self.set2ent}

    def revise(self, start_at: int = 0, save_name: str = "temp") -> list:
        """Taxonomy must be revised if an example belongs to more than one entity;
        this determination is based on the set of words that make up an example

        Args:
            start_at (int, optional): revision starting point. Defaults to 0.
            save_name (str, optional): name of saved file (no extensions). Defaults to "temp".

        Returns:
            list[dict]: history of revision changes
        """

        self.set_mappings()

        history = []
        revisions = {
            k: entities for k, entities in self.set2ent.items() if len(entities) > 1
        }
        total = len(revisions)

        for i, (k, entities) in enumerate(revisions.items()):
            if i < start_at:
                continue

            examples = self.set2wrd[k]
            options = list(range(len(entities)))

            # --- Decide where an example should be assigned
            TaxonomyReviser.log_message(i, total, entities, examples)
            index = TaxonomyReviser.handle_input(i, options)

            if index < 0:
                choice = TaxonomyReviser.UNK
                self.df_raw = self.df_raw.append([{choice: None}], ignore_index=True)
            else:
                choice = entities[options[index]]

            # --- Update dataframe
            logging.info(f"> Selected: {choice}")
            logging.info(f"CHANGES:")

            # 1. Mark examples to move
            changes = [
                {"entity": entity, "example": example, "choice": choice}
                for entity in entities
                for example in examples
                if any(self.df_raw[entity] == example)
            ]

            # 2. Remove example from all entities
            for change in changes:
                entity = change["entity"]
                example = change["example"]
                mask = self.df_raw[entity] == example
                self.df_raw[entity] = self.df_raw[entity][~mask]

            # 3. Add all examples to chosen entity
            self.df_raw = self.df_raw.append(
                [{choice: example} for example in examples], ignore_index=True
            )

            # 4. Update history
            history += [
                {"example": example, "entity": {"choices": entities, "chosen": choice}}
                for example in examples
            ]

            # 5. Display changes
            TaxonomyReviser.log_changes(changes)

            # 6. Cleanup and save
            self.df_raw = TaxonomyReviser.copy_dataframe(self.df_raw)

            if save_name:
                TaxonomyReviser.auto_save(save_name, self.df_raw, history)

        return history

    @staticmethod
    def log_message(idx: int, total: int, entities: list, examples: list):
        message = (
            f"PROGRESS: {idx} / {total}\n"
            "Choose an entity that the example should belong to:\n"
            f"- Labels: {entities}\n"
            f"- Examples: {examples}\n"
            "- Choices:\n"
        )

        logging.info(message)

    @staticmethod
    def handle_input(iteration: int, options: list):
        user_input = input(f"Enter an index (int) from {options}, -1 meaning neither: ")
        index = int(user_input)

        if index not in [-1, *options]:
            raise ValueError(
                f"ERROR! Failed at iteration {iteration}. {index} is invalid, valid choices are: {[-1, *options]}!"
            )

        return index

    @staticmethod
    def log_changes(changes):
        for change in changes:
            entity = change["entity"]
            example = change["example"]
            choice = change["choice"]
            logging.info(f"- MOVED `{example}` FROM [{entity}] TO [{choice}]")

        logging.info(
            "\n+---------------------------------------------------------------------------+\n"
        )

    @staticmethod
    def copy_dataframe(df: pd.DataFrame):
        return pd.concat(
            [
                pd.DataFrame({column: df[column].dropna().tolist()})
                for column in df.columns
            ],
            axis=1,
        )

    @staticmethod
    def auto_save(save_name: str, df: pd.DataFrame, history: list):
        df.to_csv(f"{save_name}.csv", index=False)
        json_dump(f"{save_name}.json", history)
