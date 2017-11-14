import os
import pickle
from collections import Counter
#import tensorflow as tf
import tarfile
import numpy as np
from functools import reduce
import itertools
import re
import h5py
import gc
import random
data_path = 'data/'
vocab_file = os.path.join('data/', 'vocab.h5')
directories = [ 'data/CBTest/data/train1.txt']
directories=[ 'test.txt']
choice = 5665165
testdic = 'test_noa.txt'
real_answer = 'answer.txt'
def tokenize(sentence):
    return [s.strip() for s in re.split('(\W+)+', sentence) if s.strip()]
def get_stories(story_file):
    stories = parse_stories(story_file)
    flatten = lambda story: reduce(lambda x, y: x + y, story)
    storiest = [(flatten(story), q, a,answers) for story, q, a,answers in stories]
    return stories,storiest

def parse_stories(lines):
    stories = []
    story = []
    num = 0
    flag = 0
    with open(lines, 'r') as infile:
        for line in infile:
#             num += 1
 #            print(num)
                line = line.strip()
                if not line:
                    if random.random() < choice:
                        flag = 1
                    story = []
                else:
                    _, line = line.split(' ', 1)
                    if line:
                        if '\t' in line: # query line
                            
                                q, a, answers = line.split('\t')
                                # print(q)
                                q = tokenize(q)
                                # print(q)
                                answers = answers.split('|')
      #                          print(answers)
                                # answers = tokenize(answers)

                                stories.append((story, q, a,answers))
                         
                        else:
                        
                               story.append(tokenize(line))

    return stories

def vectorize_stories(data, word2idx, doc_max_len, query_max_len):
    X = []
    Xq = []
    Y = []
    Xas = []
    for s, q, a ,answers in data:
        x = [word2idx[w] for w in s]
        xq = [word2idx[w] for w in q]
        xas = [word2idx[w] for w in answers]
        y = np.zeros(len(word2idx) + 1)
        X.append(x)
        Xq.append(xq)
        Xas.append(xas)
        Y.append(word2idx[a])

    X = pad_sequences(X, maxlen=doc_max_len)
    Q = pad_sequences(Xq, maxlen=query_max_len)
    return (X, Q, np.array(Y),Xas)

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def build_vocab(data_filenames,stories = None):
    if os.path.isfile(vocab_file):
        (word2idx, doc_length, query_length) = pickle.load( open( vocab_file, "rb" ) )
    else:
        if stories == None:
            stories = []
            for  filename in data_filenames:
                stories = stories + get_stories(filename)

        doc_length = max([len(s) for s, _, _,_ in stories])
        query_length = max([len(q) for _, q, _,_ in stories])

        print('Document Length: {}, Query Length: {}'.format(doc_length, query_length))
        vocab = sorted(set(itertools.chain(*(story + q + [answer] + answers for story, q, answer ,answers in stories))))
        vocab_size = len(vocab) + 1
        print('Vocab size:', vocab_size)
        word2idx = dict((w, i + 1) for i,w in enumerate(vocab))
        pickle.dump( (word2idx, doc_length, query_length), open( vocab_file, "wb" ) )

    return (word2idx, doc_length, query_length)  
def load_word2vec_embeddings(vocab_embed_file,save_file='embed'):
    if os.path.isfile(save_file+'.h5'):
        (W, embed_dim) = pickle.load( open( save_file+'.h5', "rb" ))
    else:
        if vocab_embed_file is None:
            return None, 384

        fp = open(vocab_embed_file, encoding='utf-8')
        word2idx, doc_length, query_length = build_vocab(directories)
        embed_dim = 384
        # vocab_embed: word --> vector
        vocab_embed = {}
        num = 0
        for line in fp:
            num += 1
            line = line.split()
       # print(len(line))
            try:
                vocab_embed[line[0]] = np.array(list(map(float, line[1:])), dtype='float32')
         #  print(vocab_embed[line[0]].shape)
            except Exception as e:
                print(e)
                continue

        fp.close()

        vocab_size = len(word2idx)
        W = np.random.randn(vocab_size, embed_dim).astype('float32')
        n = 0
        for w, i in word2idx.items():
            if w in vocab_embed:
                if vocab_embed[w].shape[0] == 384:
                    W[i, :] = vocab_embed[w]
                n += 1
        print(n)
        pickle.dump((W, embed_dim), open(save_file+'.h5', "wb"))
    return W, embed_dim

def createRecord(docs,queries,answers,xas,directory,name):
    out_name = name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(out_name)
    num = len(docs)
    print(num)
    for i in range(num):
        document = docs[i].tolist()
        query = queries[i].tolist()
        xanswers = xas[i]#.tolist()
        answer = [answers[i]]
        print(xanswers)
        example = tf.train.Example(
           features = tf.train.Features(
             feature = {
               'document': tf.train.Feature(
                 int64_list=tf.train.Int64List(value=document)),
               'query': tf.train.Feature(
                 int64_list=tf.train.Int64List(value=query)),
               'answer': tf.train.Feature(
                 int64_list=tf.train.Int64List(value=answer)),
                 'answers': tf.train.Feature(
                     int64_list=tf.train.Int64List(value=xanswers))
               }))

        serialized = example.SerializeToString()
        writer.write(serialized)

def main():

  name = ['train1','valid','test']
  name1 = [ 'test.h5']
  i=0
  XX = []
  YY = []
  dic = {}
  for dir in directories:
       if os.path.isfile(name1[i]):
           (s,stories) = pickle.load(open(name1[i], "rb"))
       else:
           s,stories = get_stories(dir)
           gc.collect()
           pickle.dump((s,stories ), open(name1[i], "wb"))
       print(len(stories))
       for s, q, a, answers in stories:

           # print(s)
           dic[''.join(str(s+q))] = a
       print('build vocab')
       num = len(dic)
       print(num)
       # for j in range(num):
       #     nX = Q[j].tolist()
       #     nX += X[j].tolist()
       #     dic[''.join(str(nX))] = Y[j]

       #createRecord(X,Q,Y,xas,dir,name[i])
       i+=1
  print('DONE')
  num = len(dic)
  print(num)

  s, stories = get_stories(testdic)
  print(len(stories))
  n = 0
  with open(real_answer, 'w')as fout:
      for s, q, a, answers in stories:
          try:
             fout.write((dic[''.join(str(s+q))] + '\n'))
             print(dic[''.join(str(s+q))])
          except Exception as e:
              fout.write('Keddah' + '\n')
              print(e)
              continue
          n+=1
  print(n)

if __name__ == "__main__":
  main()

