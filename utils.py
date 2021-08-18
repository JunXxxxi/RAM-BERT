import os
import ast
import spacy
import json
import numpy as np
import xml.etree.ElementTree as ET
from errno import ENOENT
from collections import Counter
from bert_serving.client import BertClient
import logging
from torch.utils.data import DataLoader, Dataset

# logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")
bc = BertClient()
np.set_printoptions(threshold='nan')
#
# def load_datasets_and_vocabs(FLAGS):
#     train, test = get_dataset(FLAGS.dataset_name) #json文件以list类型返回
#
#     logger.info('Train set size: %s', len(train))
#     logger.info('Test set size: %s,', len(test))
#
#     # Build word vocabulary(part of speech, dep_tag) and save pickles.
#     word_vecs, word_vocab = load_and_cache_vocabs(
#         train+test, FLAGS)
#     train_dataset = ASBA_Depparsed_Dataset(
#         train, FLAGS, word_vocab)
#     test_dataset = ASBA_Depparsed_Dataset(
#         test, FLAGS, word_vocab)
#
#     return train_dataset, test_dataset, word_vocab
#
# def get_dataset(dataset_name):
#     '''
#     Already preprocess the data and now they are in json format.(only for semeval14)
#     Retrieve train and test set
#     With a list of dict:
#     e.g. {"sentence": "Boot time is super fast, around anywhere from 35 seconds to 1 minute.",
#     "tokens": ["Boot", "time", "is", "super", "fast", ",", "around", "anywhere", "from", "35", "seconds", "to", "1", "minute", "."],
#     "tags": ["NNP", "NN", "VBZ", "RB", "RB", ",", "RB", "RB", "IN", "CD", "NNS", "IN", "CD", "NN", "."],
#     "predicted_dependencies": ["nn", "nsubj", "root", "advmod", "advmod", "punct", "advmod", "advmod", "prep", "num", "pobj", "prep", "num", "pobj", "punct"],
#     "predicted_heads": [2, 3, 0, 5, 3, 5, 8, 5, 8, 11, 9, 9, 14, 12, 3],
#     "dependencies": [["nn", 2, 1], ["nsubj", 3, 2], ["root", 0, 3], ["advmod", 5, 4], ["advmod", 3, 5], ["punct", 5, 6], ["advmod", 8, 7], ["advmod", 5, 8],
#                     ["prep", 8, 9], ["num", 11, 10], ["pobj", 9, 11], ["prep", 9, 12], ["num", 14, 13], ["pobj", 12, 14], ["punct", 3, 15]],
#     "aspect_sentiment": [["Boot time", "positive"]], "from_to": [[0, 2]]}
#     '''
#     rest_train = 'data/restaurant/Restaurants_Train_v2_biaffine_depparsed_with_energy.json'
#     rest_test = 'data/restaurant/Restaurants_Test_Gold_biaffine_depparsed_with_energy.json'
#
#     laptop_train = 'data/laptop/Laptop_Train_v2_biaffine_depparsed.json'
#     laptop_test = 'data/laptop/Laptops_Test_Gold_biaffine_depparsed.json'
#
#
#     ds_train = {'rest': rest_train,
#                 'laptop': laptop_train}
#     ds_test = {'rest': rest_test,
#                'laptop': laptop_test}
#
#     train = list(read_sentence_depparsed(ds_train[dataset_name]))
#     logger.info('# Read %s Train set: %d', dataset_name, len(train))
#
#     test = list(read_sentence_depparsed(ds_test[dataset_name]))
#     logger.info("# Read %s Test set: %d", dataset_name, len(test))
#     return train, test
#
# def read_sentence_depparsed(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#         return data
#
# def load_and_cache_vocabs(data, args):
#
#     word_vocab = None
#     word_vecs = None
#     return word_vecs, word_vocab
#
# class ASBA_Depparsed_Dataset(Dataset):
#     '''
#     Convert examples to features, numericalize text to ids.
#     data:
#         -list of dict:
#             keys: sentence, tags, pos_class, aspect, sentiment,
#                 predicted_dependencies, predicted_heads,
#                 from, to, dep_tag, dep_idx, dependencies, dep_dir
#
#     After processing,
#     data:
#         sentence
#         tags
#         pos_class
#         aspect
#         sentiment
#         from
#         to
#         dep_tag
#         dep_idx
#         dep_dir
#         predicted_dependencies_ids
#         predicted_heads
#         dependencies
#         sentence_ids
#         aspect_ids
#         tag_ids
#         dep_tag_ids
#         text_len
#         aspect_len
#         if bert:
#             input_ids
#             word_indexer
#
#     Return from getitem:
#         sentence_ids
#         aspect_ids
#         dep_tag_ids
#         dep_dir_ids
#         pos_class
#         text_len
#         aspect_len
#         sentiment
#         deprel
#         dephead
#         aspect_position
#         if bert:
#             input_ids
#             word_indexer
#             input_aspect_ids
#             aspect_indexer
#         or:
#             input_cat_ids
#             segment_ids
#     '''
#
#     def __init__(self, data, FLAGS, word_vocab):
#         self.data = data
#         self.word_vocab = word_vocab
#
#         self.convert_features()
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         e = self.data[idx]
#         items = e['dep_tag_ids'], \
#             e['pos_class'], e['text_len'], e['aspect_len'], e['sentiment'],\
#             e['dep_rel_ids'], e['predicted_heads'], e['aspect_position'], e['dep_dir_ids']
#
#         bert_items = e['input_ids'], e['word_indexer'], e['input_aspect_ids'], e['aspect_indexer'], e['input_cat_ids'], e['segment_ids']
#         # segment_id
#         items_tensor = tuple(torch.tensor(t) for t in bert_items)
#         items_tensor += tuple(torch.tensor(t) for t in items)
#         return items_tensor
#
#     def convert_features_bert(self, i):
#         """
#         BERT features.
#         convert sentence to feature.
#         """
#         cls_token = "[CLS]"
#         sep_token = "[SEP]"
#         pad_token = 0
#         # tokenizer = self.args.tokenizer
#
#         tokens = []
#         word_indexer = []
#         aspect_tokens = []
#         aspect_indexer = []
#
#         for word in self.data[i]['sentence']:
#             word_tokens = self.args.tokenizer.tokenize(word)
#             token_idx = len(tokens)
#             tokens.extend(word_tokens)
#             # word_indexer is for indexing after bert, feature back to the length of original length.
#             word_indexer.append(token_idx)
#
#         # aspect
#         for word in self.data[i]['aspect']:
#             word_aspect_tokens = self.args.tokenizer.tokenize(word)
#             token_idx = len(aspect_tokens)
#             aspect_tokens.extend(word_aspect_tokens)
#             aspect_indexer.append(token_idx)
#
#         # The convention in BERT is:
#         # (a) For sequence pairs:
#         #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#         #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
#         # (b) For single sequences:
#         #  tokens:   [CLS] the dog is hairy . [SEP]
#         #  type_ids:   0   0   0   0  0     0   0
#
#         tokens = [cls_token] + tokens + [sep_token]
#         aspect_tokens = [cls_token] + aspect_tokens + [sep_token]
#         word_indexer = [i+1 for i in word_indexer]
#         aspect_indexer = [i+1 for i in aspect_indexer]
#
#         input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
#         input_aspect_ids = self.args.tokenizer.convert_tokens_to_ids(
#             aspect_tokens)
#
#         # check len of word_indexer equals to len of sentence.
#         assert len(word_indexer) == len(self.data[i]['sentence'])
#         assert len(aspect_indexer) == len(self.data[i]['aspect'])
#
#         # THE STEP:Zero-pad up to the sequence length, save to collate_fn.
#
#         if self.args.pure_bert:
#             input_cat_ids = input_ids + input_aspect_ids[1:]
#             segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])
#
#             self.data[i]['input_cat_ids'] = input_cat_ids
#             self.data[i]['segment_ids'] = segment_ids
#         else:
#             input_cat_ids = input_ids + input_aspect_ids[1:]
#             segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])
#
#             self.data[i]['input_cat_ids'] = input_cat_ids
#             self.data[i]['segment_ids'] = segment_ids
#             self.data[i]['input_ids'] = input_ids
#             self.data[i]['word_indexer'] = word_indexer
#             self.data[i]['input_aspect_ids'] = input_aspect_ids
#             self.data[i]['aspect_indexer'] = aspect_indexer
#
#     def convert_features(self):
#         '''
#         Convert sentence, aspects, pos_tags, dependency_tags to ids.
#         '''
#         for i in range(len(self.data)):
#             if self.args.embedding_type == 'glove':
#                 self.data[i]['sentence_ids'] = [self.word_vocab['stoi'][w]
#                                                 for w in self.data[i]['sentence']]
#                 self.data[i]['aspect_ids'] = [self.word_vocab['stoi'][w]
#                                               for w in self.data[i]['aspect']]
#             elif self.args.embedding_type == 'elmo':
#                 self.data[i]['sentence_ids'] = self.data[i]['sentence']
#                 self.data[i]['aspect_ids'] = self.data[i]['aspect']
#             else:  # self.args.embedding_type == 'bert'
#                 self.convert_features_bert(i)
#
#             self.data[i]['text_len'] = len(self.data[i]['sentence'])
#             self.data[i]['aspect_position'] = [0] * self.data[i]['text_len']
#             try:  # find the index of aspect in sentence
#                 for j in range(self.data[i]['from'], self.data[i]['to']):
#                     self.data[i]['aspect_position'][j] = 1
#             except:
#                 for term in self.data[i]['aspect']:
#                     self.data[i]['aspect_position'][self.data[i]
#                                                     ['sentence'].index(term)] = 1
#
#             self.data[i]['dep_tag_ids'] = [self.dep_tag_vocab['stoi'][w]
#                                            for w in self.data[i]['dep_tag']]
#             self.data[i]['dep_dir_ids'] = [idx
#                                            for idx in self.data[i]['dep_dir']]
#             self.data[i]['pos_class'] = [self.pos_tag_vocab['stoi'][w]
#                                              for w in self.data[i]['tags']]
#             self.data[i]['aspect_len'] = len(self.data[i]['aspect'])
#
#             self.data[i]['dep_rel_ids'] = [self.dep_tag_vocab['stoi'][r]
#                                            for r in self.data[i]['predicted_dependencies']]


#####################################################################


# def trans(p):
#     words = []  # 建立一个空列表
#     index = 0  # 遍历所有的字符
#     start = 0  # 记录每个单词的开始位置
#     while index < len(p):  # 当index小于p的长度
#         start = index  # start来记录位置
#         while p[index] != " " and p[index] not in [".", ","]:  # 若不是空格，点号，逗号
#             index += 1  # index加一
#             if index == len(p):  # 若遍历完成
#                 break  # 结束
#         words.append(p[start:index])
#         if index == len(p):
#             break
#         while p[index] == " " or p[index] in [".", ","]:
#             if p[index] in [".", ","]:
#                 words.append(p[index:index+1])
#             index += 1
#             if index == len(p):
#                 break
#     return words



def get_inputs(sentence, aspects, tokenizer):
    """
            BERT features.
            convert sentence to feature.
            """
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = 0
    # tokenizer = self.args.tokenizer

    tokens = []
    word_indexer = []
    aspect_tokens = []
    aspect_indexer = []


    for word in sentence:
        word_tokens = tokenizer.tokenize(word)
        token_idx = len(tokens)
        tokens.extend(word_tokens)
        # word_indexer is for indexing after bert, feature back to the length of original length.
        word_indexer.append(token_idx)

    n = len(tokens)
    # aspect
    for word in aspects:
        word_aspect_tokens = tokenizer.tokenize(word)
        token_idx = n+len(aspect_tokens)
        aspect_tokens.extend(word_aspect_tokens)
        aspect_indexer.append(token_idx)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0

    tokens = [cls_token] + tokens + [sep_token]
    aspect_tokens = [cls_token] + aspect_tokens + [sep_token]
    word_indexer = [i + 1 for i in word_indexer]
    aspect_indexer = [i + 2 for i in aspect_indexer]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_aspect_ids = tokenizer.convert_tokens_to_ids(
        aspect_tokens)

    # check len of word_indexer equals to len of sentence.
    assert len(word_indexer) == len(sentence)
    assert len(aspect_indexer) == len(aspects)

    # THE STEP:Zero-pad up to the sequence length, save to collate_fn.

    input_cat_ids = input_ids + input_aspect_ids[1:]
    segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

    return input_cat_ids,input_aspect_ids,segment_ids,word_indexer,aspect_indexer

def get_data_info(train_fname, test_fname, save_fname, pre_processed):
    word2id, max_sentence_len, max_aspect_len = {}, 0, 0
    word2id['<pad>'] = 0
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        with open(save_fname, 'r') as f:
            for line in f:
                content = line.strip().split()
                if len(content) == 3:
                    max_sentence_len = int(content[1])
                    max_aspect_len = int(content[2])
                else:
                    word2id[content[0]] = int(content[1])
    else:
        if not os.path.isfile(train_fname):
            raise IOError(ENOENT, 'Not a file', train_fname)
        if not os.path.isfile(test_fname):
            raise IOError(ENOENT, 'Not a file', test_fname)

        words = []

        train_tree = ET.parse(train_fname)
        train_root = train_tree.getroot()
        for sentence in train_root:
            sptoks = nlp(sentence.find('text').text)
            words.extend([sp.text.lower() for sp in sptoks])
            if len(sptoks) > max_sentence_len:
                max_sentence_len = len(sptoks)
            for asp_terms in sentence.iter('aspectTerms'):
                for asp_term in asp_terms.findall('aspectTerm'):
                    if asp_term.get('polarity') == 'conflict':
                        continue
                    t_sptoks = nlp(asp_term.get('term'))
                    if len(t_sptoks) > max_aspect_len:
                        max_aspect_len = len(t_sptoks)
        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word:
                word2id[word] = len(word2id)
    
        test_tree = ET.parse(test_fname)
        test_root = test_tree.getroot()
        for sentence in test_root:
            sptoks = nlp(sentence.find('text').text)
            words.extend([sp.text.lower() for sp in sptoks])
            if len(sptoks) > max_sentence_len:
                max_sentence_len = len(sptoks)
            for asp_terms in sentence.iter('aspectTerms'):
                for asp_term in asp_terms.findall('aspectTerm'):
                    if asp_term.get('polarity') == 'conflict':
                        continue
                    t_sptoks = nlp(asp_term.get('term'))
                    if len(t_sptoks) > max_aspect_len:
                        max_aspect_len = len(t_sptoks)
        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word:
                word2id[word] = len(word2id)

        with open(save_fname, 'w') as f:
            f.write('length %s %s\n' % (max_sentence_len, max_aspect_len))
            for key, value in word2id.items():
                f.write('%s %s\n' % (key, value))
                
    print('There are %s words in the dataset, the max length of sentence is %s, and the max length of aspect is %s' % (len(word2id), max_sentence_len, max_aspect_len))
    return word2id, max_sentence_len, max_aspect_len

def get_loc_info(sptoks, from_id, to_id):
    aspect = []
    for sptok in sptoks:
        if sptok.idx < to_id and sptok.idx + len(sptok.text) > from_id:
            aspect.append(sptok.i)
    loc_info = []
    for _i, sptok in enumerate(sptoks):
        loc_info.append(min([abs(_i - i) for i in aspect]) / len(sptoks))
    return loc_info


def get_inputs2(tokens_, aspect_tokens_,s,a):
    # s = 0
    # a = 0
    doc_vecs = bc.encode([tokens_,aspect_tokens_])
    # for i in doc_vecs[0]:
    #     if np.all(i == 0):
    #         break
    #     else:
    #         s = s + 1
    # for j in doc_vecs[1]:
    #     if np.all(j == 0):
    #         break
    #     else:
    #         a = a + 1
    sentence = np.random.normal(0, 0.05, [s, 768])
    aspect = np.random.normal(0, 0.05, [a, 768])
    for n in range(0, s):
        u = str(doc_vecs[0][n]).strip('[]')
        u = u.split()
        v = np.array(list(map(float, u[0:])))
        sentence[n] = v
    for n in range(0, a):
        u = str(doc_vecs[1][n]).strip('[]')
        u = u.split()
        v = np.array(list(map(float, u[0:])))
        aspect[n] = v
    return sentence,aspect



def read_data(fname, word2id, max_sentence_len, max_aspect_len, save_fname, pre_processed):
    sentences, aspects, sentence_lens, sentence_locs, labels = [], [], [], [], []
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        lines = open(save_fname, 'r').readlines()
        for i in range(0, len(lines), 5):
            sentences.append(ast.literal_eval(lines[i]))
            aspects.append(ast.literal_eval(lines[i + 1]))
            sentence_lens.append(ast.literal_eval(lines[i + 2]))
            sentence_locs.append(ast.literal_eval(lines[i + 3]))
            labels.append(ast.literal_eval(lines[i + 4]))
    else:
        if not os.path.isfile(fname):
            raise IOError(ENOENT, 'Not a file', fname)

        tree = ET.parse(fname)
        root = tree.getroot()
        tokens_ = []
        aspect_tokens_ = []
        segment_ids = []
        word_indexer = []
        aspect_indexer = []
        sentence_ = []
        aspect_ = []
        sentence_vec = []
        aspect_vec = []
        with open(save_fname, 'w') as f:
            for sentence in root:
                sptoks = nlp(sentence.find('text').text)
                sptoks2 = sentence.find('text').text
                if len(sptoks.text.strip()) != 0:
                    ids = []
                    for sptok in sptoks:
                        if sptok.text.lower() in word2id:
                            ids.append(word2id[sptok.text.lower()])
                    for asp_terms in sentence.iter('aspectTerms'):
                        for asp_term in asp_terms.findall('aspectTerm'):
                            if asp_term.get('polarity') == 'conflict':
                                continue
                            t_sptoks = nlp(asp_term.get('term'))
                            t_sptoks2 = asp_term.get('term')
                            t_ids = []
                            for sptok in t_sptoks:
                                if sptok.text.lower() in word2id:
                                    t_ids.append(word2id[sptok.text.lower()])
                            tokens = [token for token in sptoks]
                            aspect_tokens = [token for token in t_sptoks]

                            # sentence_.append(sptoks2)
                            # f.write("%s\n" % sentence_[-1])
                            # aspect_.append(t_sptoks2)
                            # f.write("%s\n" % aspect_[-1])
                            # for i in tokens:
                            #     tokens_.append(str(i))
                            # for i in aspect_tokens:
                            #     aspect_tokens_.append(str(i))
                            # tokens,aspect_tokens,segment_id,word_index,aspect_index = get_inputs(tokens_,aspect_tokens_,tokenizer)
                            s_vec,a_vec = get_inputs2(sptoks2,t_sptoks2,max_sentence_len,max_aspect_len)
                            sentence_vec.append(s_vec)
                            f.write("%s\n" % sentence_vec[-1])
                            aspect_vec.append(a_vec)
                            f.write("%s\n" % aspect_vec[-1])
                            # sentences.append(tokens)
                            # f.write("%s\n" % sentences[-1])
                            # segment_ids.append(segment_id)
                            # f.write("%s\n" % segment_ids[-1])
                            # word_indexer.append(word_index)
                            # f.write("%s\n" % word_indexer[-1])
                            # aspect_indexer.append(aspect_index)
                            # f.write("%s\n" % aspect_indexer[-1])
                            # aspects.append(aspect_tokens)
                            # f.write("%s\n" % aspects[-1])
                            sentence_lens.append(len(sptoks))
                            f.write("%s\n" % sentence_lens[-1])
                            loc_info = get_loc_info(sptoks, int(asp_term.get('from')), int(asp_term.get('to')))
                            sentence_locs.append(loc_info + [1] * (max_sentence_len - len(loc_info)))
                            f.write("%s\n" % sentence_locs[-1])
                            polarity = asp_term.get('polarity')
                            if polarity == 'negative':
                                labels.append([1, 0, 0])
                            elif polarity == 'neutral':
                                labels.append([0, 1, 0])
                            elif polarity == "positive":
                                labels.append([0, 0, 1])
                            f.write("%s\n" % labels[-1])

    print("Read %s sentences from %s" % (len(sentence_vec), fname))
    # return np.asarray(sentences),np.asarray(segment_ids),np.asarray(word_indexer),np.asarray(aspect_indexer),np.asarray(aspects), np.asarray(sentence_lens), np.asarray(sentence_locs), np.asarray(labels)
    return np.asarray(sentence_vec), np.asarray(aspect_vec), np.asarray(sentence_lens), np.asarray(sentence_locs), np.asarray(labels)

def load_word_embeddings(fname, embedding_dim, word2id):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)

    word2vec = np.random.normal(0, 0.05, [len(word2id), embedding_dim])
    oov = len(word2id)
    with open(fname, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            content = line.strip().split(' ')
            if content[0] in word2id:
                word2vec[word2id[content[0]]] = np.array(list(map(float, content[1:])))
                oov = oov - 1
    word2vec[word2id['<pad>'], :] = 0
    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec

def get_batch_index(length, batch_size, is_shuffle=True):
    index = list(range(length))
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
        yield index[i * batch_size:(i + 1) * batch_size]
