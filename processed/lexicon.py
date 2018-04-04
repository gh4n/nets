# -*- coding: utf-8 -*-
# Build lexicon search structure to constrain inference

# Usage:
# lex = Lexicon('lexicon.h5', verbosity=0, radius=3)
# lex.get_candidates('autoencoder', lex.dictionary)

import time
import pandas as pd
from pyxdameraulevenshtein import damerau_levenshtein_distance

class Lexicon(object):

    def __init__(self, lexicon_path, radius, verbosity):
        
        self.lexicon_path = lexicon_path
        self.verbosity = verbosity
        self.radius = radius
        self.longest_word_length = 0
        self.dictionary = {}

        print('Building lexicon search structure ...')
        start = time.time()
        self.create_dictionary()
        startup = time.time() - start
        print('-----\nStartup: {:.3f} s\n-----'.format(startup))    


    def get_deletes_list(self, w):
        '''Given a word, derive predictions with up to <radius> characters deleted'''
        deletes = []
        queue = [w]
        for d in range(self.radius):
            temp_queue = []
            for word in queue:
                if len(word)>1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c+1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue
        
        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries: (list of suggested corrections, frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1]==1:
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list 
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added


    def create_dictionary(self):
        unique_word_count = 0
        lex = pd.read_hdf(self.lexicon_path)
        total_word_count = lex.shape[0]
        words = lex['word'].values

        for word in words:
            if self.create_dictionary_entry(word):
                unique_word_count += 1

        print("Total words in lexicon: %i" % total_word_count)
        print("Unique words in lexicon: %i" % unique_word_count)
        print("Total items in lexicon (corpus words and deletions): %i" % len(self.dictionary))
        print("  Edit distance for deletions: %i" % self.radius)
        print("  Length of longest word in corpus: %i" % self.longest_word_length)

        # return self.dictionary

    def get_candidates(self, prediction, dictionary, silent=False):
        # Constrain prediciton to nearest-neighbour candidates in lexicon
        from collections import deque
        
        if (len(prediction) - self.longest_word_length) > self.radius:
            if not silent:
                print("No items in lexicon within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = deque([prediction])
        q_dictionary = {}  # items other than prediction that we've checked

        while len(queue)>0:
            q_item = queue.popleft()

            # early exit
            if ((self.verbosity<2) and (len(suggest_dict)>0) and
                  ((len(prediction)-len(q_item))>min_suggest_len)):
                break

            # process queue item
            if (q_item in dictionary) and (q_item not in suggest_dict):
                if (dictionary[q_item][1]>0):
                    assert len(prediction)>=len(q_item)
                    suggest_dict[q_item] = (dictionary[q_item][1],
                                            len(prediction) - len(q_item))
                    # early exit
                    if ((self.verbosity<2) and (len(prediction)==len(q_item))):
                        break
                    elif (len(prediction) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(prediction) - len(q_item)

                # the suggested corrections for q_item as stored in 
                # dictionary (whether or not q_item itself is a valid word 
                # or merely a delete) can be valid corrections
                for sc_item in dictionary[q_item][0]:
                    if (sc_item not in suggest_dict):

                        # suggested items should always be longer 
                        # (unless manual corrections are added)
                        assert len(sc_item)>len(q_item)

                        # q_items that are not input should be shorter 
                        # than original prediction 
                        # (unless manual corrections added)
                        assert len(q_item)<=len(prediction)

                        if len(q_item)==len(prediction):
                            assert q_item==prediction
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as 
                        # the prediction itself
                        assert sc_item!=prediction

                        # calculate edit distance 
                        item_dist = damerau_levenshtein_distance(sc_item, prediction)

                        # do not add words with greater edit distance if 
                        # self.verbosity setting not on
                        if ((self.verbosity<2) and (item_dist>min_suggest_len)):
                            pass
                        elif item_dist<=self.radius:
                            assert sc_item in dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words 
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if self.verbosity
                        # setting not on
                        if self.verbosity<2:
                            suggest_dict = {k:v for k, v in list(suggest_dict.items()) if v[1]<=min_suggest_len}

            # now generate deletes (e.g. a subprediction of prediction or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(prediction)>=len(q_item)

            # do not add words with greater edit distance if self.verbosity setting 
            # is not on
            if ((self.verbosity<2) and ((len(prediction)-len(q_item))>min_suggest_len)):
                pass
            elif (len(prediction)-len(q_item))<self.radius and len(q_item)>1:
                for c in range(len(q_item)): # character index        
                    word_minus_c = q_item[:c] + q_item[c+1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to 
        # list for output
        if not silent and self.verbosity!=0:
            print("number of possible corrections: %i" %len(suggest_dict))
            print("  edit distance for deletions: %i" % self.radius)

        # output option 1
        # sort results by ascending order of edit distance and descending 
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x: 
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction, 
        #                                  (frequency in corpus, edit distance)):
        as_list = list(suggest_dict.items())
        outlist = sorted(as_list, key=lambda term_freq_dist: (term_freq_dist[1][1], -term_freq_dist[1][0]))

        if self.verbosity==0:
            try:
                return outlist[0]
            except IndexError:
                print('No match within {} single character edits found within lexicon'.format(self.radius))
                return (prediction, None)
        else:
            return outlist



