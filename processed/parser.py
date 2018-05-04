import csv
import string
import re
import pandas as pd
import operator
from sklearn.model_selection import train_test_split


def clean_str(str):
    """
    Removes whitespace, punctuation and strings of numbers
    """
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    str = regex.sub('', str)
    str = re.sub(r'[0-9]{3}\w+', 'NUM', str)
    str = re.sub(r' +', ' ', str)
    return str


def spell_check(word):

    return


def process(file, categories):
    x, y = [], []
    with open(file, newline='') as f:
        data = csv.DictReader(f)
        for row in data:
            category = clean_str(row['Category'].lower())
            desc = clean_str(row['short_description'].lower())
            if not category:
                category_enum = categories["misc"]
            else:
                category_enum = categories[category]
            x.append(desc)
            y.append(category_enum)
    save(x, y, file+'_clean', False)
    return x, y


def save(x, y, name, df=True):
    data = pd.DataFrame({'x': x, 'y': y})
    if df:
        return data
    else:
        return data.to_csv(name)


def get_freqs():
    return



def split_data(split=[80, 20, 20]):
    x_train, x_test1, y_train, y_test1 = train_test_split(x, y, test_size=0.2, random_state=42)
    return



if __name__ == "__main__":
    print(clean_str("HELLO asdfasdf   9999999"))