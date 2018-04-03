import csv
import string
import re
import matplotlib.pyplot as plt

freqs = {}
categories = {"misc":0, "access issues security enablement": 1, "application": 2, "hw":3, "job failures":4, "nw":5, "sw":6}
regex = re.compile('[%s]' % re.escape(string.punctuation))

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

        # create frequency table of words in description
        for word in desc_tokens:
            if word not in freqs:
                freqs[word] = 1
            else:
                freqs[word] += 1

        desc_tokens.append(category_enum)


