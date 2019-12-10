import nltk
import random
import string
import sqlite3
import warnings
warnings.simplefilter("ignore", UserWarning)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


conn = sqlite3.connect('../../data/reddit_comments.db')
#conn = sqlite3.connect('../../data/question_answer.db')
c = conn.cursor()


c.execute("SELECT parent, comment FROM parent_reply WHERE subreddit = 'funny'")
#c.execute("SELECT question, answer FROM question_answer")


inp = [(parent.lower(), comment.lower()) for (parent, comment) in c.fetchall() if parent is not None and comment is not None]


reference_sentences = []
for p, _ in inp:
    reference_sentences.append(p)

print(reference_sentences)

wnlemmatizer = nltk.stem.WordNetLemmatizer()

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
    epsilon_response = ''
    reference_sentences.append(user_input)

    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(reference_sentences)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]
    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]


    if vector_matched == 0:
        epsilon_response = epsilon_response + "I am sorry, I could not understand you"
        return epsilon_response
    else:
        epsilon_response = epsilon_response + str(inp[similar_sentence_number][1])
        return epsilon_response


def run_bot():
    continue_dialogue = True
    print("Hello, I am your friend Epsilon. You can ask me anything:")

    while(continue_dialogue == True):
        human_text = input()
        human_text = human_text.lower()
        if human_text != 'bye':
            if human_text == 'thanks' or human_text == 'thank you very much' or human_text == 'thank you':
                continue_dialogue = False
                print("Epsilon: Most welcome")
            else:
                if generate_greeting_response(human_text) != None:
                    print("Epsilon: " + generate_greeting_response(human_text))
                else:
                    print("Epsilon: ", end="")
                    print(generate_response(human_text))
                    reference_sentences.remove(human_text)
        else:
            continue_dialogue = False
            print("Epsilon: Good bye and take care of yourself...")

if __name__ == '__main__':
    run_bot()