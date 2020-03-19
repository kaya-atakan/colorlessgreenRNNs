import torch
import numpy as np

# X = torch.arange(12).view(4, 3)
# surprisal_subset = mask.X.cpu().numpy().astype(bool) #Boolean mask

# mask = torch.zeros((4, 3))

sents = open("generated.text", "r").readlines()
golds = open('generated.eval', "r").readlines() #indices of targets

# print(len(sents)) # 410
# print(len(golds)) # 410
#TODO optimize by initializaing np.array of needed size and doing indexing
targets = []

# print(list(zip(sents, golds)))

index_col = 0
for sent, gold in zip(sents, golds):
    # print(sent)
    target_idx = int(gold.split()[index_col])
    len_s = len(sent.split(" ")) #length of the sentences
    t_s = [0] * len_s # array of zeros being equal to the len of sentences
    t_s[target_idx] = 1 #targets are marked as 1 and the remain is all zeros
    targets.extend(t_s)

print(np.array(targets))
# mask_data = torch.LongTensor(np.array(targets))

def batchify(data, bsz):
    # bsz = Batch Size
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
   
    """ If you’re more familiar with advanced indexing, for a 2D tensor t2d, 
    t2d.narrow(1, 0, 10) is the same as t2d[:, 0:10] 
    and t2d.narrow(1, 5, 2) is the same as t2d[:, 5:7].
    narrow is interesting as for higher dimensionnal tensors, you don’t have to do : for every other dimension. 
    Also narrow() (like select()) will always work inplace: it does not use any extra memory, 
    is very fast and any modification of the narrowed tensor will impact the original one. """
    
    data = torch.narrow(data, 0, 0, nbatch * bsz)

    # Evenly divide the data across the bsz batches.
    
    """ reshape() can operate on both contiguous and non-contiguous tensor 
    while view() can only operate on contiguous tensor."""
    
    """ we can find that transpose(), narrow() and tensor slicing, and expand() will make the generated tensor not contiguous
    The answer is it the view() function cannot be applied to a discontiguous tensor. 
    This is probably because view() requires that the tensor to be contiguously stored so that it can do fast reshape in memory."""
    data = data.view(bsz, -1).t().contiguous()
    # if cuda:
    #     data = data.cuda()
    return data

# mask_data = batchify(torch.LongTensor(mask), eval_batch_size)

eval_batch_size = 1
# nbatch = mask_data.size(0) // eval_batch_size

# mask_data_narrowed = torch.narrow(mask_data, 0, 0, nbatch * eval_batch_size)
# print(mask_data_narrowed)

# print(mask_data_narrowed.shape)
# print(mask_data.view(eval_batch_size, -1).shape) # 11950 columns
# print(mask_data.view(eval_batch_size, -1).t().shape) # 11950 rows

# print(mask_data.view(eval_batch_size, -1).t().contiguous())

aaa = torch.Tensor( [[1,2,3],[4,5,6]] )

# print(aaa.shape)

# print(batchify(mask_data, 1))


# Create tensor from 0 to 10.
main_data = torch.arange(50001, dtype=torch.long)
# print(main_data // 1)
print(batchify(main_data, eval_batch_size).shape)

import os
import torch
from collections import defaultdict
import logging

class Dictionary(object):
    # abs_path = '/Users/atakan/Desktop/Git_folder/colorlessgreenRNNs'
    def __init__(self, path):
        """ parameters::
        path is the path to test file"""
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)

        vocab_path = os.path.join(path, 'vocab.txt')
        print(vocab_path)
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            # print(vocab)
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            # print(self.word2idx)
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        
        except FileNotFoundError:
            logging.info("Vocab file not found, creating new vocab file.")
            self.create_vocab(os.path.join(path, 'train.txt'))
            open(vocab_path,"w").write("\n".join([w for w in self.idx2word]))

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1


    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)


    def __repr__(self):
        return "<Test a:%s b:%s>" % (self.word2idx, self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary(path)
        self.train = tokenize(self.dictionary, os.path.join(path, 'train.txt'))
        self.valid = tokenize(self.dictionary, os.path.join(path, 'valid.txt'))
        self.test = tokenize(self.dictionary, os.path.join(path, 'test.txt'))


def tokenize(dictionary, path):
    """Tokenizes a text file for training or testing to a sequence of indices format
       We assume that training and test data has <eos> symbols """
    assert os.path.exists(path)
    with open(path, 'r', encoding="utf8") as f:
        ntokens = 0
        for line in f:
            words = line.split()
            ntokens += len(words)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        ids = torch.LongTensor(ntokens)
        token = 0
        for line in f:
            words = line.split() #Split gives you a list for a each word
            for word in words:
                if word in dictionary.word2idx:
                    ids[token] = dictionary.word2idx[word]
                else:
                    ids[token] = dictionary.word2idx["<unk>"]
                token += 1

    return ids

foo = Dictionary('/Users/atakan/Desktop/Git_folder/colorlessgreenRNNs/data/agreement/English')
# foo.__len__() #50001
# print(foo.create_vocab('/Users/atakan/Desktop/Git_folder/colorlessgreenRNNs/data/agreement/English/vocab.txt'))

# print(tokenize(foo, '/Users/atakan/Desktop/Git_folder/colorlessgreenRNNs/data/agreement/English/vocab.txt'))

for i in range(0, main_data.size(0) - 1, 20):
    print(i)