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

def get_df(file):
    """
    """
    return

def get_csv(file):
    """

    """
    return




if __name__ == "__main__":
    print(clean_str("HELLO asdfasdf   9999999"))