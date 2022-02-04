# -*- coding:utf-8 -*-

import re
import pickle
import unicodedata
from gensim import corpora
from nltk import word_tokenize


def to_words(sentence):
    sentence_list = [re.sub(r"(\w+)(!+|\?+|…+|\.+|,+|~+)", r"\1", word) for word in sentence.split(' ')]
    return sentence_list


def is_english(string):
    for ch in string:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            return False
        if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return False
    return True


class ConvCorpus:
    def __init__(self, file_path, batch_size=100, size_filter=False):
        self.posts = []
        self.cmnts = []
        self.dic = None

        print("###########################")
        print("File path : ",file_path)
        print("###########################")
        print()

        if file_path is not None:
            print("###########################")
            print("construct dict is called")
            print("###########################")
            print()
            self._construct_dict(file_path, batch_size, size_filter)

    def _construct_dict(self, file_path, batch_size, size_filter):
        # define sentence and corpus size
        max_length = 20

        # preprocess
        # this stores all tokenized lists of words
        posts = []
        cmnts = []
        pattern = '(.+?)(\t)(.+?)(\n|\r\n)'
        # (.+?)(\t)(.+?)(\n|\r\n)
        # /
        # gm
        # 1st Capturing Group (.+?)
        # . matches any character (except for line terminators)
        # +? matches the previous token between one and unlimited times, as few times as possible, expanding as needed (lazy)
        # 2nd Capturing Group (\t)
        # \t matches a tab character (ASCII 9)
        # 3rd Capturing Group (.+?)
        # . matches any character (except for line terminators)
        # +? matches the previous token between one and unlimited times, as few times as possible, expanding as needed (lazy)
        # 4th Capturing Group (\n|\r\n)
        # 1st Alternative \n
        # \n matches a line-feed (newline) character (ASCII 10)
        # 2nd Alternative \r\n
        # \r matches a carriage return (ASCII 13)
        # \n matches a line-feed (newline) character (ASCII 10)
        # Global pattern flags
        # g modifier: global. All matches (don't return after first match)
        # m modifier: multi line. Causes ^ and $ to match the begin/end of each line (not only begin/end of string)
        r = re.compile(pattern)

        for index, line in enumerate(open(file_path, 'r', encoding='utf-8')):
            # Scan through string looking for the first location where the regular expression pattern produces a match, and return a corresponding match object. Return None if no position in the string matches the pattern; note that this is different from finding a zero-length match at some point in the string.
            m = r.search(line)
            if m is not None:
                #group method returns the complete matched subgroup by default or a tuple of matched subgroups depending on the number of arguments
                # group 1 contains question
                # group 3 contains answer
                if is_english(m.group(1) + m.group(3)):
                    # tokenizing the words
                    post = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(m.group(1))]
                    cmnt = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(m.group(3))]
                    if size_filter:
                        if len(post) <= max_length and len(cmnt) <= max_length:
                            posts.append(post)
                            cmnts.append(cmnt)
                    else:
                        posts.append(post)
                        cmnts.append(cmnt)

        # cut corpus for a batch size
        remove_num = len(posts) - (int(len(posts) / batch_size) * batch_size)
        # removing extra lines of conversation to make it round figure in hundreds
        # 12768 -> 12700
        # print(remove_num)
        del posts[len(posts)-remove_num:]
        del cmnts[len(cmnts)-remove_num:]
        print(len(posts), 'of pairs has been collected!')

        # construct dictionary
        # creating dictionary from these conversation
        self.dic = corpora.Dictionary(posts + cmnts, prune_at=None)

        # no_below : int, optional
        #     Keep tokens which are contained in at least `no_below` documents.
        # no_above : float, optional
        #     Keep tokens which are contained in no more than `no_above` documents
        #     (fraction of total corpus size, not an absolute number).
        # keep_n : int, optional
        #     Keep only the first `keep_n` most frequent tokens.
        self.dic.filter_extremes(no_below=1, no_above=1.0, keep_n=10000)
        print(len(self.dic))
        # add symbols
        print("index of start : ",len(self.dic.token2id))
        self.dic.token2id['<start>'] = len(self.dic.token2id)
        print("index of eos : ",len(self.dic.token2id))
        self.dic.token2id['<eos>'] = len(self.dic.token2id)
        print("index of unk : ",len(self.dic.token2id))
        self.dic.token2id['<unk>'] = len(self.dic.token2id)
        self.dic.token2id['<pad>'] = -1

        # make ID corpus
        # storing tokenid lists of list
        self.posts = [[self.dic.token2id.get(word, self.dic.token2id['<unk>']) for word in post] for post in posts]
        self.cmnts = [[self.dic.token2id.get(word, self.dic.token2id['<unk>']) for word in cmnt] for cmnt in cmnts]

    def save(self, save_dir):
        self.dic.save(save_dir + 'dictionary.dict')
        with open(save_dir + 'posts.list', 'wb') as f:
            pickle.dump(self.posts, f)
        with open(save_dir + 'cmnts.list', 'wb') as f:
            pickle.dump(self.cmnts, f)

    def load(self, load_dir):
        self.dic = corpora.Dictionary.load(load_dir + 'dictionary.dict')
        with open(load_dir + 'posts.list', 'rb') as f:
            self.posts = pickle.load(f)
        with open(load_dir + 'cmnts.list', 'rb') as f:
            self.cmnts = pickle.load(f)
        print(len(self.posts), 'of pairs has been collected!')
