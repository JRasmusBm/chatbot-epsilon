import re
import string
import json
import sqlite3


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

def insert(t):
  sql = ''' INSERT INTO question_answer(question,answer)
                VALUES (?,?)'''
  c.execute(sql, t)
  conn.commit()

if __name__ == '__main__':
  create_connection(r"/chatbot-epsilon/data/question_answer.db")
  conn = sqlite3.connect('question_answer.db')
  c = conn.cursor()
  create_table()
  '''with open('dev-v2.0.json', 'r') as file:
    data = json.load(file)

    for overhead in data['data']:
      for qas in overhead['paragraphs']:
        for qa in qas['qas']:
          if len(qa['answers']) > 0:
            print(qa['question'], '\n', qa['answers'][0]['text'])
            try:
              insert((str(qa['question']), str(qa['answers'][0]['text'])))
            except:
              pass'''

  with open('dev-v2.0.json', 'r') as file:
    data = json.load(file)

    for overhead in data['data'][-2:-1]:
      print(overhead)
      for qas in overhead['paragraphs']:
        for qa in qas['qas']:
          print(qa)
          if len(qa['answers']) > 0:
            print(qa['question'], '\n', qa['answers'][0]['text'])