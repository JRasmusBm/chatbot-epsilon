import nltk
#nltk.download('punkt')
import random
import string
import sqlite3


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


conn = sqlite3.connect('reddit_comments.db')
c = conn.cursor()


c.execute("SELECT parent, comment FROM parent_reply WHERE subreddit='funny'")


inp = [(parent.lower(), comment.lower()) for (parent, comment) in c.fetchall() if parent is not None and comment is not None]
#print(input[600])


reference_text = ''
reference_sentences = []
for p, _ in inp:
    reference_sentences.append(p)
    reference_text += p




#reference_text = re.sub(r'\[[0-9]*\]', ' ', reference_text)
#reference_text = re.sub(r'\s+', ' ', reference_text)


corpus = nltk.word_tokenize(reference_text)


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
    Redditor_response = ''
    reference_sentences.append(user_input)

    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(reference_sentences)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]
    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        Redditor_response = Redditor_response + "I am sorry, I could not understand you"
        return Redditor_response
    else:
        Redditor_response = Redditor_response + str(inp[similar_sentence_number][1])
        return Redditor_response

word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
all_word_vectors = word_vectorizer.fit_transform(reference_sentences)

similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)

similar_sentence_number = similar_vector_values.argsort()[0][-2]

continue_dialogue = True
print("Hello, I am your friend Redditor. You can ask me anything:")

while(continue_dialogue == True):
    human_text = input()
    human_text = human_text.lower()
    if human_text != 'bye':
        if human_text == 'thanks' or human_text == 'thank you very much' or human_text == 'thank you':
            continue_dialogue = False
            print("Redditor: Most welcome")
        else:
            if generate_greeting_response(human_text) != None:
                print("Redditor: " + generate_greeting_response(human_text))
            else:
                print("Redditor: ", end="")
                print(generate_response(human_text))
                reference_sentences.remove(human_text)
    else:
        continue_dialogue = False
        print("Redditor: Good bye and take care of yourself...")