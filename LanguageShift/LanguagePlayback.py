import pandas as pd

from LanguageShift.LanguageModel import LanguageModel


class LanguagePlayback(LanguageModel):
    def __init__(self):
        super()

    def read_file(self, filename):
        return pd.read_csv(filename).dropna()
