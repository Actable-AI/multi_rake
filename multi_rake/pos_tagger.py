from flair.models import SequenceTagger
from flair.data import Sentence


class MultilingualPOSTagger(object):

    SUPPORTED_LANGUAGE_CODES = ["en", "fr", "de", "it", "nl", "pl", "es", "sv", "da",
                                "no", "fi", "cs"]

    def __init__(self,
                 flair_pos_model_path=None,
                 flair_pos_supported_language_codes=None):
        self.flair_tagger = SequenceTagger.load(flair_pos_model_path or 'pos-multi')
        self.supported_language_codes = \
            flair_pos_supported_language_codes or self.SUPPORTED_LANGUAGE_CODES

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

