import csv
import string
import re
import pandas as pd
import operator
import matplotlib.pyplot as plt
# import seaborn as sns

freqs = {}
categories = {"misc":0, "access issues security enablement": 1, "application": 2, "hw":3, "job failures":4, "nw":5, "sw":6}
regex = re.compile('[%s]' % re.escape(string.punctuation))
x = []
y = []

with open('data/stop-word-list.csv') as f:
    data = csv.reader(f)

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
x = [num for num in range(len(occs))]
#plt.margins(0.05, 0.1)
#plt.barh(x, occs, align='center', alpha=0.5)
# sns.distplot(occs[:200], bins=50, kde=False)
print(occs)
plt.show()





stupid_dict =  {'desc':x, 'cat':y}

df = pd.DataFrame(stupid_dict)
print(df)
