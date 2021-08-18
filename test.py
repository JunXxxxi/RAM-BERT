from errno import ENOENT
import os
import ast
from bert_serving.client import BertClient
import numpy as np
np.set_printoptions(suppress=True)
bc = BertClient()
s = 0
a = 0
sentence_vet = []

doc_vecs = bc.encode(['First do it it it it it it it it it it it it it it it it it it it it it it it it.', 'jacksonvill'])
for i in doc_vecs[0]:
    if np.all(i==0):
        break
    else:
        s = s+1
for j in doc_vecs[1]:
    if np.all(j==0):
        break
    else:
        a = a+1
sentence = np.random.normal(0, 0.05, [s, 768])
aspect = np.random.normal(0, 0.05, [a, 768])
for n in range(0,s):
    u = str(doc_vecs[0][n]).strip('[]')
    u = u.split()
    v = np.array(list(map(float, u[0:])))
    sentence[n] = v
for n in range(0,a):
    u = str(doc_vecs[1][n]).strip('[]')
    u = u.split()
    v = np.array(list(map(float, u[0:])))
    aspect[n] = v
# sentence = np.array(sentence)
# aspect = np.array(aspect)
print(sentence)
print(aspect)
sentence_vet.append(sentence)
sentence_vet.append(aspect)
np.asarray(sentence_vet)
print(len(sentence_vet))


print(sentence.size)
print(doc_vecs[1][4])
print(np.all(doc_vecs[1][4]==0))
print(doc_vecs[0][1])
print(sentence[1])


# sentence_vec, aspect_vec, sentence_lens, sentence_locs, labels = [], [], [], [], []
# if not os.path.isfile("data/train_data.txt"):
#     raise IOError(ENOENT, 'Not a file data/train_data.txt')
# lines = open("data/train_data.txt", 'r').readlines()
# for i in range(0, len(lines), 5):
#     sentence_vec.append(ast.literal_eval(lines[i]))
#     aspect_vec.append(ast.literal_eval(lines[i + 1]))
#     sentence_lens.append(ast.literal_eval(lines[i + 2]))
#     sentence_locs.append(ast.literal_eval(lines[i + 3]))
#     labels.append(ast.literal_eval(lines[i + 4]))
# print(sentence_vec)
# print("Read %s sentences from %s" % (len(sentence_vec),"train_data.txt"))
    # return np.asarray(sentences),np.asarray(segment_ids),np.asarray(word_indexer),np.asarray(aspect_indexer),np.asarray(aspects), np.asarray(sentence_lens), np.asarray(sentence_locs), np.asarray(labels)
