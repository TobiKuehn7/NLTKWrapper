import nltk
from nltk import PunktSentenceTokenizer, word_tokenize, ngrams, jaccard_distance
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer


class NLTKWrapper:

    text = ''

    def __init__(self, copora='wordnet', text=''):
        self.text = text.replace('\n', ' ')
        nltk.download('punkt')
        nltk.download('tagsets')
        nltk.download('vader_lexicon')
        nltk.download('omw-1.4')
        nltk.download('words')
        nltk.download(copora)

    def _check_arguments(self, text):
        """
        Checks if the passed argument and the text variable are empty. If they are the function raises a ValueError.
        If the argument text varibale is not empty it will be set as class variable.
        :param text: the input argument to check
        """
        if text == '':
            if self.text == '':
                raise ValueError('You have to pass a text or set the `self.text` variable of this class to tokenize a '
                                 'text')
        else:
            self.text = text.replace('\n', ' ')

    def word_tokenization(self, text=''):
        self._check_arguments(text)
        tokenized_text = word_tokenize(self.text)
        return tokenized_text

    def sentence_tokenization(self, text=''):
        self._check_arguments(text)
        pst = PunktSentenceTokenizer()
        tokenized_text = pst.tokenize(self.text)

        return tokenized_text

    def pos_tagging(self, text='', tagset=''):
        self._check_arguments(text)
        if tagset == '':
            pos_tags = nltk.pos_tag(word_tokenize(self.text))
        else:
            pos_tags = nltk.pos_tag(word_tokenize(self.text), tagset=tagset)

        return pos_tags

    def sentiment_classification(self, text=''):
        self._check_arguments(text)
        sentences = []
        sentences.extend(self.sentence_tokenization(self.text))
        classifications_list = []
        for sentence in sentences:
            sentence_classification = [sentence]
            sid = SentimentIntensityAnalyzer()
            ss = sid.polarity_scores(sentence)
            for k in sorted(ss):
                sentence_classification.append(f'{k}: {ss[k]}')

            classifications_list.append(sentence_classification)

        return classifications_list

    def word_frequencies(self, text=''):
        self._check_arguments(text)
        w_frequencies = nltk.FreqDist(word_tokenize(self.text))
        return w_frequencies.most_common()

    def n_grams(self, n, text=''):
        self._check_arguments(text)
        n_gram = ngrams(self.text.split(), n)

        n_grams = []
        for grams in n_gram:
            n_grams.append(grams)

        return n_grams

    def lemmatization(self, text=''):
        self._check_arguments(text=text)
        wnl = WordNetLemmatizer()
        lemmatized_words = []
        for word in self.word_tokenization(self.text):
            lemmatized_words.append(wnl.lemmatize(word))

        return lemmatized_words

    def spelling_correction(self, text=''):
        self._check_arguments(text)
        correct_spellings = nltk.corpus.words.words()

        words = self.word_tokenization(self.text)
        corrected_words = []

        for word in words:
            temp = [(jaccard_distance(set(ngrams(word, 1)), set(ngrams(w, 1))), w)
                    for w in correct_spellings if w[0] == word[0]]
            corrected_words.append(sorted(temp, key=lambda val: val[0])[0][1])

        return corrected_words

    def show_tags(self, tagset_name):
        """
        prints the definitions from the tagset. If tagset name == all, all tagsets definitions will be printed.
        :param tagset_name: possible inputs:
            1. all -> all tagsets definitions will be printed
            2. upenn -> prints upenn tagset
            3. claws5 -> prints claws5 tagset
            4. brown -> prints brown tagset\n
            everything else will raise a ValueError
        """
        if tagset_name == 'all':
            print(f'upenn_tagset:\n{nltk.help.upenn_tagset()}')
            print(f'claws5_tagset:\n{nltk.help.claws5_tagset()}')
            print(f'brown_tagset:\n{nltk.help.brown_tagset()}')
        elif tagset_name == 'upenn':
            print(f'upenn_tagset:\n{nltk.help.upenn_tagset()}')
        elif tagset_name == 'claws5':
            print(f'claws5_tagset:\n{nltk.help.claws5_tagset()}')
        elif tagset_name == 'upenn':
            print(f'brown_tagset:\n{nltk.help.brown_tagset()}')
        else:
            raise ValueError('you have to pass a valid input. check doc for valid inputs.')
