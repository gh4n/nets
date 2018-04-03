import csv
import string
import re
import pandas as pd
import operator
from sklearn.model_selection import train_test_split


freqs = {}
categories = {"misc":0, "access issues security enablement": 1, "application": 2, "hw":3, "job failures":4, "nw":5, "sw":6}
regex = re.compile('[%s]' % re.escape(string.punctuation))
x = []
y = []


with open('/home/han/Projects/accenture/nets/data/deepdive-bootcamp.csv', newline='') as f:
    data = csv.DictReader(f)
    for row in data:
        category = row['Category'].lower()
        desc = row['short_description'].lower()

        # remove punctuation
        # strip spaces
        category_clean = regex.sub('', category)
        desc_clean = regex.sub('', desc)
        category_clean = re.sub(r'[0-9]{3}\w+', 'NUM', category_clean)
        desc_clean = re.sub(r'[0-9]{3}\w+', 'NUM', desc_clean)
        desc_clean = re.sub(r' +', ' ', desc_clean)
        category_clean = re.sub(r' +',' ', category_clean)
        desc_tokens = desc_clean.split(' ')

        if not category_clean:
            category_enum = categories["misc"]
        else:
            category_enum = categories[category_clean]

        x.append(desc_clean)
        y.append(category_enum)


        # create frequency table of words in description
        for word in desc_tokens:
            if word not in freqs:
                freqs[word] = 1
            else:
                freqs[word] += 1

freqs_sorted = sorted(freqs.items(), key=operator.itemgetter(1), reverse=True)
word, occs= zip(*freqs_sorted)

df_dict = {'words':word, 'occs':occs}
freqs_df = pd.DataFrame(df_dict)




stupid_dict =  {'desc':x, 'cat':y}

df = pd.DataFrame(stupid_dict)
x_train, x_test1, y_train, y_test1 = train_test_split(x, y, test_size=0.2, random_state=42)

x_test, x_val, y_test, y_val = train_test_split(x_test1, y_test1, test_size=0.5, random_state=42)

def save_df(x, y, name):
    df = pd.DataFrame({'text': x, 'category': y})
    df.to_csv(name)


save_df(x_train, y_train, 'deepdive_train.csv')
save_df(x_test, y_test, 'deepdive_test.csv')
save_df(x_val, y_val, 'deepdive_val.csv')