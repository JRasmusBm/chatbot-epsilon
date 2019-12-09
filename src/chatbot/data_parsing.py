import re
import string
import json
import sqlite3

with open('dev-v2.0.json', 'r') as file:
    data = json.load(file)
    for overhead in data['data'][-2:-1]:
      for qas in overhead['paragraphs']:
        for qa in qas['qas']:
          if len(qa['answers']) > 0:
            print('q:', qa['question'],'\na:', qa['answers'][0]['text'], '\n')
      #print(k['paragraphs']['question'])


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))


def create_connection(db_file):
  """ create a database connection to a SQLite database """
  conn = None
  try:
    conn = sqlite3.connect(db_file)
    print(sqlite3.version)
  except sqlite3.Error as e:
    print(e)
  finally:
    if conn:
      conn.close()

def create_table():
  c.execute('CREATE TABLE IF NOT EXISTS question_answer(question TEXT PRIMARY KEY, answer TEXT)')

def insert(question, answer):
  sql = ''' INSERT INTO 'question_answer'('question','answer')
                VALUES(?,?) '''
  c.execute(sql, (question, answer))
  return c.lastrowid

if __name__ == '__main__':
  create_connection(r"C:\Users\J\Documents\GitHub\chatbot-epsilon\src\chatbot\question_answer.db")
  conn = sqlite3.connect('question_answer.db')
  c = conn.cursor()
  create_table()
  print(insert('test', 'test'))
