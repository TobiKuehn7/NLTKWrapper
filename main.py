from wrapper import NLTKWrapper

if __name__ == '__main__':
    """
    Run this code and see the resuts of all wrapper functions.
    """
    # the testtext file holds the text which is analysed in this
    with open('testtext.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # initalize the wrapper with the text from the 'testtext.txt' file
    nw = NLTKWrapper(text=text)
    # word tokenization
    w_tokens = nw.word_tokenization()
    print(f'Word tokenization: \n{w_tokens}')
    # sentence tokenization
    s_tokens = nw.sentence_tokenization()
    print(f'Sentence tokenization: \n{s_tokens}')
    # part of speech (pos) tagging
    pos_tags = nw.pos_tagging()
    print(f'Part of speech tagging: \n{pos_tags}')
    # part of speech (pos) tagging with universal tagset
    pos_tags_universal = nw.pos_tagging(tagset='universal')
    print(f'Part of speech tagging with tagset universal: \n{pos_tags_universal}')
    # sentiment classification
    classifications = nw.sentiment_classification()
    print(f'Sentiment classification: \n{classifications}')
    # word frequencies
    word_frequencies = nw.word_frequencies()
    print(f'Word frequencies: \n{word_frequencies}')
    # TODO: sentence frequencies
    # n-grams
    n = 3
    n_grams = nw.n_grams(n)
    print(f'n-grams with n = {n}: \n{n_grams}')
    n = 6
    n_grams = nw.n_grams(n)
    print(f'n-grams with n = {n}: \n{n_grams}')
    # lemmatization
    lemmatized_words = nw.lemmatization()
    print(f'Lemmatized words: \n{lemmatized_words}')
    # spelling correction
    corrected_words = nw.spelling_correction('I catn tyep')
    print(f'Corrected Words: \n{corrected_words}')

    # show tags (uncomment the line below to see the tags of all tagsets, or replace 'all' with a tagsets name. check
    # the documentation of the function for available tagsets)
    # nw.show_tags('all')
