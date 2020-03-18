# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dictionary_corpus
from utils import repackage_hidden, batchify, get_batch
import numpy as np

parser = argparse.ArgumentParser(description='Mask-based evaluation: extracts softmax vectors for specified words')

parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str,
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--path', type=str, help='path to test file (text) gold file (indices of words to evaluate)')
parser.add_argument('--suffix', type=str, help='suffix for generated output files which will be saved as path.output_suffix')
args = parser.parse_args()


def evaluate(data_source, mask):

    """ paramaters::     
    data_source is the test_data, which is a tensor of shape ([50001, 1])
    
    mask is an np.array (11950,) coming from create_target_mask function below
     """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    # print(len(data_source))
    print(model)
    
    """ The answer lies in init_hidden. It is not the hidden layer weights but the initial hidden state in RNN/LSTM, 
    which is h0 in the formulas. 
   
    For every epoch, we should re-initialize a new beginner hidden state, this is because during the testing, 
    our model will have no information about the test sentence and will have a zero initial hidden state."""
    
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, seq_len):
        # keep continuous hidden state across all sentences in the input file
        data, targets = get_batch(data_source, i, seq_len, evaluation=True) #data = list of tensors of indices
        _, targets_mask = get_batch(mask, i, seq_len, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, vocab_size)
        total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets).data
        # print(total_loss.shape())
        # print(total_loss.item())
        
        # print(targets)
        # print(targets_mask) #Put 1 for the target, 0 for the other words
        
        output_candidates_probs(output_flat, targets, targets_mask)

        hidden = repackage_hidden(hidden)

    # return total_loss[0] / (len(data_source) - 1)
    return total_loss.item() / (len(data_source) - 1)


def output_candidates_probs(output_flat, targets, mask):
    log_probs = F.log_softmax(output_flat).data
    surprisal = -1 * log_probs
    # print('Surprisal scores is LOADING')
    # print(surprisal)

    # print('-------------')
    log_probs_np = log_probs.cpu().numpy()
    surprisal_np = surprisal.cpu().numpy()

    # print(surprisal_np)
    
    subset = mask.data.cpu().numpy().astype(bool)
    surprisal_subset = mask.data.cpu().numpy().astype(bool) #Boolean mask
    # print(surprisal_subset)

    idx2word = dictionary.idx2word #len = 50001, #type = list of lists

    # print(idx2word)


    # print(targets.data.cpu().numpy()) # This returns the index of each word in the sentences
    
    # print(targets.data.cpu().numpy()[surprisal_subset]) #This returns the indices of each target in sentences

    # for scores, correct_label in zip(log_probs_np[subset], targets.data.cpu().numpy()[subset]):
    
  
    # for scores, correct_label in zip(surprisal_np[surprisal_subset], targets.data.cpu().numpy()[surprisal_subset]):
    #     print(idx2word[correct_label], scores[correct_label])

        
        # f_output.write("\t".join(str(s) for s in scores) + "\n")


def create_target_mask(test_file, gold_file, index_col):
    """ This function returns one np array
        where all the targets  in each sentence are marked as 1. The rest of the 
        sentence is marked as 0  """
    sents = open(test_file, "r").readlines() #test sentences, generated.text
    golds = open(gold_file, "r").readlines() #indices of targets, generated.eval
    
    #TODO optimize by initializaing np.array of needed size and doing indexing
    targets = []
    for sent, gold in zip(sents, golds):
        # constr_id, sent_id, word_id, pos, morph
        # print(int(gold.split()[index_col]))
        target_idx = int(gold.split()[index_col]) #Gets target indices
        len_s = len(sent.split(" ")) #gets the length of sentences
        t_s = [0] * len_s
        t_s[target_idx] = 1
        #print(sent.split(" ")[target_idx])
        targets.extend(t_s)
    return np.array(targets)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    print("Loading the model")
    if args.cuda:
        model = torch.load(f)
    else:
        # to convert model trained on cuda to cpu model
        model = torch.load(f, map_location = lambda storage, loc: storage)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

eval_batch_size = 1
seq_len = 20

dictionary = dictionary_corpus.Dictionary(args.data)
# print(dictionary)
vocab_size = len(dictionary)
print("Vocab size", vocab_size)
print("TESTING")

# assuming the mask file contains one number per line indicating the index of the target word
index_col = 0

mask = create_target_mask(args.path + ".text", args.path + ".eval", index_col) #which takes generated.text and .eval as input
mask_data = batchify(torch.LongTensor(mask), eval_batch_size, args.cuda)
test_data = batchify(dictionary_corpus.tokenize(dictionary, args.path + ".text"), eval_batch_size, args.cuda)

# print(test_data.shape)

# f_output = open(args.path + ".output_" + args.suffix, 'w')
# evaluate(test_data, mask_data)
# f_output.close()


