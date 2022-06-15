from flair.models import SequenceTagger
from flair.data import Sentence


class MultilingualPOSTagger(object):

    SUPPORTED_LANGUAGE_CODES = ["en", "fr", "de", "it", "nl", "pl", "es", "sv", "da",
                                "no", "fi", "cs"]

    def __init__(self):
        self.flair_tagger = SequenceTagger.load('pos-multi')

    def tag(self, sentence_raw):
        sentence = Sentence(sentence_raw)
        self.flair_tagger.predict(sentence)
        return [
            {
                "start_position": t.start_position,
                "end_position": t.end_position,
                "word": sentence_raw[t.start_position:t.end_position],
                "pos": t.get_label().value,
            } for t in sentence.tokens
        ]

