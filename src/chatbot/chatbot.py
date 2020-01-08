import random
import re
import string
import urllib.request

import bs4 as bs
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.simplefilter("ignore", UserWarning)

#nltk.download('punkt')
#nltk.download("wordnet")

sports_html = bs.BeautifulSoup(urllib.request.urlopen("https://en.wikipedia.org/wiki/List_of_sports").read(), "lxml")
global article_sentences, wnlemmatizer

def is_valid(url):
    return bool(url) and "#" not in url and ":" not in url and '/wiki/' == url[:6]
possible_subjects = {v.get('href'): v.get_text() for v in sports_html.find_all('a') if is_valid(v.get('href'))}


def change_sport(user_input):
    subjects = [k for k in possible_subjects.values()]
    subjects.append(user_input)

    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_subject_vectors = word_vectorizer.fit_transform(subjects)
    similar_vector_values = cosine_similarity(all_subject_vectors[-1], all_subject_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]
    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    subjects.remove(user_input)

    if vector_matched == 0:
        return "I am sorry, I don't know anything about that subject. Are you sure that " + user_input + " is a sport?"
    else:
        for k, v in possible_subjects.items():
            if v == subjects[similar_sentence_number]:
                print(k)
                read_corpus(k)
                break
        return "I have now learned about " + user_input + "! Ask me something :)"

def read_corpus(subject):
    raw_html = urllib.request.urlopen("https://en.wikipedia.org" + subject)

    raw_html = raw_html.read()
    article_html = bs.BeautifulSoup(raw_html, "lxml")

    article_paragraphs = article_html.find_all("p")

    article_text = ""

    for para in article_paragraphs:
        article_text += para.text

    article_text = article_text.lower()

    article_text = re.sub(r"\[[0-9]*\]", " ", article_text)
    article_text = re.sub(r"\s+", " ", article_text)
    global  article_sentences
    article_sentences = nltk.sent_tokenize(article_text)
    global wnlemmatizer
    wnlemmatizer = nltk.stem.WordNetLemmatizer()

read_corpus('/wiki/Sport')

def perform_lemmatization(tokens):
    return [wnlemmatizer.lemmatize(token) for token in tokens]

punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)

def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))

greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup", "hello")
greeting_responses = ["hey", "hey hows you?", "*nods*", "hello, how you doing", "hello", "Welcome, I am good and you"]

def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)


def generate_response(user_input):
    article_sentences.append(user_input)

    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(article_sentences)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]
    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    article_sentences.remove(user_input)

    if vector_matched == 0:
        return "I am sorry, I could not understand you"
    else:
        return str(article_sentences[similar_sentence_number])


def run_bot():
    print('Hello, I am your friend Epsilon. You can ask me anything about sport! To change subject type "change subject" :)')

    while True:
        human_text = input()
        human_text = human_text.lower()
        if human_text != 'bye':
            if human_text == 'change subject':
                print('Epsilon: What subject do you want to talk about?')
                subject = input()
                print(change_sport(subject).capitalize())
            else:
                if generate_greeting_response(human_text) != None:
                    print("Epsilon: " + generate_greeting_response(human_text))
                else:
                    print("Epsilon: ", end="")
                    print(generate_response(human_text).capitalize())
        else:
            print("Epsilon: Good bye and take care of yourself...")
            break

if __name__ == '__main__':
    run_bot()