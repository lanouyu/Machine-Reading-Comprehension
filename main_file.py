import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os, sys, time
import logging
import gc

import data_process as dp 
import GAReader

''' parser '''
parser = argparse.ArgumentParser()
parser.add_argument('--net', required=True, help='the network or model')
parser.add_argument('--data_path', default='../dataset', help='the path of data files(train/dev/test)')
parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
parser.add_argument('--save_vocab', action='store_true', help='save vocab to exp folder')
parser.add_argument('--load_model', action='store_true', help='save vocab to exp folder')
parser.add_argument('--save_model', action='store_true', help='save vocab to exp folder')
parser.add_argument('--fix_emb', action='store_true', help='fix the embed during training')
parser.add_argument('--use_feature', action='store_true', help='use qe-comm feature')
parser.add_argument('--use_char', action='store_true', help='use the char level embedding')
parser.add_argument('--deviceId', type=int, default=-1, help='train model on ith gpu. -1:cpu')
parser.add_argument('--random_seed', type=int, default=999, help='set initial random seed')
parser.add_argument('--experiment', default='exp', help='Where to store samples and models')
parser.add_argument('--read_model', required=False, help='read model from this file')
parser.add_argument('--init_emb', type=str, default=None, help='initalize the word embedding with this file')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate at each non-recurrent layer')

parser.add_argument('--emb_size', type=int, default=100, help='word embedding dimension')
parser.add_argument('--char_emb_dim', type=int, default=100, help='char embedding dimension')
parser.add_argument('--gru_dim', type=int, default=100, help='gru dimension')
parser.add_argument('--max_epoch', type=int, default=50, help='max number of epochs to train for')
parser.add_argument('--num_layers', type=int, default=2, help='number of hidden layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--test_batchSize', type=int, default=0, help='input batch size in decoding')
parser.add_argument('--max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
parser.add_argument('--optim', default='adam', help='choose an optimizer')
parser.add_argument('--ga_fn', default='mul', help='gated attention function for combining d & q')

opt = parser.parse_args()
opt.test_batchSize = opt.batchSize if opt.test_batchSize == 0 else opt.test_batchSize
exp_name = 'net_'+opt.net+'__lr_'+str(opt.lr)+'__char_'+str(opt.use_char)+'__feat_'+str(opt.use_feature)\
	+'__grudim_'+str(opt.gru_dim)+'__embdim_'+str(opt.emb_size)+'__opt_'+str(opt.optim)
if opt.init_emb:
	exp_name += '__init_emb_True'
else:
	exp_name += '__init_emb_False'
exp_path = os.path.join(opt.experiment, exp_name)
if not os.path.exists(exp_path):
	os.makedirs(exp_path)

''' logger '''
logFormatter = logging.Formatter('%(message)s')
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
# file out
if opt.testing:
	fileHandler = logging.FileHandler('%s/log_test.txt' % (exp_path), mode='w')
else:
	fileHandler = logging.FileHandler('%s/log_train.txt' % (exp_path), mode='w')
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
# std out
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.info(opt)
logger.info("Experiment path: %s" % (exp_path))
logger.info(time.asctime(time.localtime(time.time())))

''' GPU '''
if opt.deviceId >= 0:
	opt.cuda = True
	torch.cuda.set_device(opt.deviceId)
	logger.info("GPU %d is manually selected." % (opt.deviceId))
else:
	opt.cuda = False
	logger.info("CPU is used.")

''' random seed '''
random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
if torch.cuda.is_available():
    if not opt.cuda:
        logger.info("WARNING: You have a CUDA device, so you should probably run with --deviceId [1|2|3]")
    else:
        torch.cuda.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)

''' data process '''
logger.info('data process...')
try:
	vocab = dp.Vocab()
	opt.train_data_file = os.path.join(opt.data_path, 'train.txt')
	opt.valid_data_file = os.path.join(opt.data_path, 'dev.txt')
	opt.test_data_file = os.path.join(opt.data_path, 'test.txt')

	if opt.init_emb is not None:
		word2vec, word_size, opt.emb_size = vocab.read_vocab_file(opt.init_emb, use_cuda=opt.cuda)
	else:
		word2vec = None
		word_size = 0
	vocab_size = vocab.read_vocab_from_data([opt.train_data_file, opt.valid_data_file, opt.test_data_file])
	logger.info("Vocab size: %s" % vocab_size)
	if opt.save_vocab:
		vocab.save_vocab(os.path.join(exp_path, 'vocab.txt'))
	opt.train_data = vocab.read_data_from_file(opt.train_data_file, use_cuda=opt.cuda)
	opt.valid_data = vocab.read_data_from_file(opt.valid_data_file, use_cuda=opt.cuda)
	opt.test_data = vocab.read_data_from_file(opt.test_data_file, use_cuda=opt.cuda)
except KeyboardInterrupt:
	logger.info('KeyboardInterrupt')
	train_data.close()
	valid_data.close()
	test_data.close()

''' build model '''
if opt.net == 'GAReader':
	model = GAReader.GAReader(opt.num_layers, vocab, vocab_size, char_size=0, dropout=opt.dropout, gru_dim=opt.gru_dim, 
		word2vec=word2vec, init_word_size=word_size, fix_emb=opt.fix_emb, embed_dim=opt.emb_size, char_emb_dim=opt.char_emb_dim, char_gru_dim=50,
		use_cuda=opt.cuda, use_feature=opt.use_feature, use_char=opt.use_char, ga_fn=opt.ga_fn)
else:
	pass
loss_function = nn.CrossEntropyLoss(size_average=False)
if opt.cuda:
	model = model.cuda()
	loss_function = loss_function.cuda()
if opt.optim.lower() == 'sgd':
	optimizer = optim.SGD(model.parameters(), lr=opt.lr)
elif opt.optim.lower() == 'adam':
	optimizer = optim.Adam(mode.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
if opt.read_model:
	model.load_model(opt.read_model)

def train(train_data_batch):
	file_used = False
	best_a = 0
	best_res = {}
	for i in xrange(opt.max_epoch):
		start_time = time.time()
		losses, accuracy = [], []
		model.train()
		pointer = 0
		len_train_data_batch = len(train_data_batch)
		while pointer < len_train_data_batch:
			# load a batch of data
			pointer_next = pointer + opt.batchSize if pointer+opt.batchSize<len_train_data_batch else len_train_data_batch
			data_batch = train_data_batch[pointer:pointer_next]
			
			# train
			optimizer.zero_grad()
			probs, pred_ans = model(input_batch)
			loss = loss_function(probs.contiguous(), input_batch['answer_idx'])
			#print 'pred:', probs, 'ans:', input_batch['answer_idx']
			print 'loss', loss.data
			losses.append(loss.data)
			loss.backward()
			if opt.max_norm > 0:
				torch.nn.utils.clip_grad_norm(model.parameters(), opt.max_norm)
			optimizer.step()
		
		logger.info('Training:\tEpoch : %d\tTime : %.4fs\t[la] Loss : %.5f ' 
			% (i, time.time() - start_time, np.mean(losses, axis=0)))
		gc.collect()
		data_file.close()

		# evaluate
		model.eval()
		start_time = time.time()
		loss_val, p_val, r_val, f_val, a_val = evaluate(opt.valid_data)
		logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tLoss : %.5f\tFscore : %.5f\tAccuracy: %.5f' % (i, time.time() - start_time, loss_val, f_val, a_val))
		gc.collect()
		start_time = time.time()
		loss_te, p_te, r_te, f_te, a_val = evaluate(opt.test_data, out_path=os.path.join(exp_path, 'test_iter'+i))
		logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tLoss : %.5f\tFscore : %.5f\tAccuracy: %.5f' % (i, time.time() - start_time, loss_te, f_te, a_te))
		gc.collect()

		if best_a < a_val:
			model.save_model(os.path.join(exp_path, 'model'))
			best_a = a_val
			logger.info('NEW BEST:\tEpoch : %d\tbest valid a : %.5f;\ttest a : %.5f' % (i, a_val, a_te))
			best_res['iter'] = i
			best_res['vp'], best_res['vr'], best_res['vf'], best_res['va'] = p_val, r_val, f_val, a_val
			best_res['tp'], best_res['tr'], best_res['tf'], best_res['ta'] = p_te, r_te, f_te, a_te
	logger.info('BEST RESULT: \tEpoch : %d\tbest valid accuracy: %.5f\tbest test accuracy: %.5f' % (best_res['iter'], best_res['va'], best_res['ta']))

def evaluate(eval_data_batch, out_path=None):
	losses = []
	TP, FP, FN, TN, T, F = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	data_file = open(data_path, "rb")
	if out_path is not None:
		out_file = open(out_path, 'wb')
		out_file.write('answer\tprediction\n')
	pointer = 0
	len_eval_data_batch = len(eval_data_batch)
	while pointer < len_eval_data_batch:
		# load a batch of data
		pointer_next = pointer + opt.batchSize if pointer+opt.batchSize<len_train_data_batch else len_train_data_batch
		data_batch = train_data_batch[pointer:pointer_next]
		
		# forward
		probs, pred_ans = model(input_batch)
		loss = loss_function(probs.contiguous(), input_batch['answer_idx'])
		losses.append(loss)

		# p/r/f/a
		for idx, ans in enumerate(input_batch['answer_idx']):
			if ans == pred_ans[idx]:
				T += 1
			else:
				F += 1

		if T+F == 0:
			p, r, f, a = 0, 0, 0, 0
		else:
			#p, r, f = 100*TP/(TP+FP), 100*TP/(TP+FN), 100*2*TP/(2*TP+FN+FP)
			p, r, f = 0, 0, 0
			a = 100*T/(T+F)

		if out_file is not None:
			for i in xrange(opt.batchSize):
				answer = str(vocab.get_word(input_batch['answer'].data[i]))
				prediction = str(vocab.get_word(input_batch['cands'][i].data[pred_ans.data.squeeze(1)[i]]))
				out_file.write(answer+'\t'+prediction+'\n')
	out_file.close()
	loss = np.mean(losses, axis=0)
	print loss
	return loss[0], p, r, f, a

if opt.testing:
	logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
	model.eval()
	start_time = time.time()
	loss_te, p_te, r_te, f_te, a_te = evaluate(opt.test_data, out_path=os.path.join(exp_path, 'test'))
	logger.info('Evaluation:\tTime : %.4fs\tLoss : %.5f\tFscore : %.5f\tAccuracy: %.5f' % (time.time() - start_time, loss_te, f_te, a_te))
else:
	logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
	model.train()
	train(opt.train_data)








