import pycld2
import regex

LETTERS_RE = regex.compile(r'\p{L}+')

SENTENCE_DELIMITERS_RE = regex.compile(
    r'[\.,;:¡!¿\?…⋯‹›«»\\"“”\[\]\(\)⟨⟩}{&]'  # any punctuation sign or &
    r'|\s[-–~]+\s',  # or '-' between spaces
    regex.VERBOSE,
)

class TextSegment(object):
    """ A text segment is a text with metadata of its location.

    """
    def __init__(self, text, sentence_id, start_position, end_position):
        # end position is inclusive
        self.text = text
        self.sentence_id = sentence_id
        self.start_position = start_position
        self.end_position = end_position

    def __eq__(self, other):
        return self.text == other.text

    def __hash__(self):
        return hash(self.text)


def detect_language(text, proba_threshold):
    _, _, details = pycld2.detect(text)

    language = details[0][0].lower()
    language_code = details[0][1]
    probability = details[0][2]

    if language_code != 'un' and probability > proba_threshold:
        return language_code, language

    return None, None


def keep_only_letters(string):
    return ' '.join(token.group() for token in LETTERS_RE.finditer(string))


def separate_words(text):
    words = []

    for word in text.split():
        if not word.isnumeric():
            words.append(word)

    return words


def split_sentences(text):
    sentences = SENTENCE_DELIMITERS_RE.split(text)
    return sentences
