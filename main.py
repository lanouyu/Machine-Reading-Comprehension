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
parser.add_argument('--train_file', default='train.txt', help='the name of training data files')
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
parser.add_argument('--dropout', type=float, default=0.4, help='dropout rate at each non-recurrent layer')

parser.add_argument('--emb_size', type=int, default=100, help='word embedding dimension')
parser.add_argument('--char_emb_dim', type=int, default=25, help='char embedding dimension')
parser.add_argument('--gru_dim', type=int, default=100, help='gru dimension')
parser.add_argument('--char_gru_dim', type=int, default=50, help='char gru dimension')
parser.add_argument('--max_epoch', type=int, default=10, help='max number of epochs to train for')
parser.add_argument('--num_layers', type=int, default=3, help='number of hidden layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--test_batchSize', type=int, default=0, help='input batch size in decoding')
parser.add_argument('--max_norm', type=float, default=10, help="threshold of gradient clipping (2-norm)")
parser.add_argument('--optim', default='adam', help='choose an optimizer')
parser.add_argument('--ga_fn', default='mul', help='gated attention function for combining d & q')

opt = parser.parse_args()
opt.test_batchSize = opt.batchSize if opt.test_batchSize == 0 else opt.test_batchSize
exp_name = 'net_'+opt.net+'__lr_'+str(opt.lr)+'__char_'+str(opt.use_char)+'__feat_'+str(opt.use_feature)\
	+'__grudim_'+str(opt.gru_dim)+'__chargrudim_'+str(opt.char_gru_dim)+'__embdim_'+str(opt.emb_size)+'__charembdim_'+str(opt.char_emb_dim)+'__opt_'+str(opt.optim)
if opt.init_emb:
	exp_name += '__initemb_True'
else:
	exp_name += '__initemb_False'
if opt.fix_emb:
	exp_name += '__fixemb_True'
else:
	exp_name += '__fixemb_False'
exp_name += '__maxnorm_'+str(opt.max_norm) + '__numlayers_'+str(opt.num_layers) + '__dropout_'+str(opt.dropout)
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
	opt.train_data_file = os.path.join(opt.data_path, opt.train_file)
	opt.valid_data_file = os.path.join(opt.data_path, 'dev.txt')
	opt.test_data_file = os.path.join(opt.data_path, 'test.txt')

	if opt.init_emb is not None:
		word2vec, opt.word_size, opt.emb_size, opt.char_size = vocab.read_vocab_file(opt.init_emb, use_cuda=opt.cuda)
	else:
		word2vec = None
		opt.word_size = 0
		opt.char_size = 0
	opt.vocab_size, opt.sent_size, opt.char_size = vocab.read_vocab_from_data([opt.train_data_file, opt.valid_data_file, opt.test_data_file])
	logger.info("Vocab size: %s" % opt.vocab_size)
	logger.info("Sent size: %s" % opt.sent_size)
	if opt.save_vocab:
		vocab.save_vocab(os.path.join(exp_path, 'vocab.txt'))
	#train_data = open(train_data_file, "rb")
	#valid_data = open(train_data_file, "rb")
	#test_data = open(train_data_file, "rb")
	#train_data = vocab.read_data_in_batch(train_data_file, use_cuda=opt.cuda)
	#valid_data = vocab.read_data_in_batch(valid_data_file, use_cuda=opt.cuda)
	#test_data = vocab.read_data_in_batch(test_data_file, use_cuda=opt.cuda)

	''' build model '''
	if opt.net == 'GAReader':
		model = GAReader.GAReader(opt.num_layers, vocab, opt.vocab_size, char_size=opt.char_size, dropout=opt.dropout, gru_dim=opt.gru_dim, 
			word2vec=word2vec, init_word_size=opt.word_size, fix_emb=opt.fix_emb, embed_dim=opt.emb_size, char_emb_dim=opt.char_emb_dim, char_gru_dim=opt.char_gru_dim,
			use_cuda=opt.cuda, use_feature=opt.use_feature, use_char=opt.use_char, ga_fn=opt.ga_fn)
	else:
		pass
	#loss_function = nn.CrossEntropyLoss(size_average=False)
	loss_function = nn.NLLLoss(size_average=False)
	if opt.cuda:
		model = model.cuda()
		loss_function = loss_function.cuda()
	if opt.optim.lower() == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
	elif opt.optim.lower() == 'adam':
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
	if opt.read_model:
		model.load_model(opt.read_model)
except KeyboardInterrupt:
	logger.info('KeyboardInterrupt')


def train(data_path):
	file_used = False
	best_va = 0
	best_ta = 0
	best_vloss = 100
	best_res = {}
	for i in xrange(opt.max_epoch):
		if i >= 2:
			for p in optimizer.param_groups:
				p['lr'] /= 2
		start_time = time.time()
		losses, accs = [], []
		model.train()
		data_file = open(data_path, "rb")
		file_used = False
		idx_all = 0
		idx_prop = 0.0
		while not file_used:
			# load a batch of data
			data_batch = []
			idx = 0
			while idx < opt.batchSize:
				data_line = data_file.readline()
				if data_line == '':
					file_used = True
					break
				elif data_line == '\n':
					idx += 1
				data_batch.append(data_line)
			if data_batch != []:
				input_batch = vocab.read_data_in_batch(data_batch, idx, use_cuda=opt.cuda, use_char=opt.use_char)
			else:
				break
			# train
			#probs, pred_ans = model(input_batch)
			#loss = loss_function(probs.contiguous(), input_batch['answer_idx'])
			loss, acc, pred_ans = model(input_batch)
			'''
			print 'probs:', probs.data
			print 'pred:', pred_ans.data.t()
			print 'ans:', input_batch['answer_idx'].data.unsqueeze(1).t()
			'''
			losses.append(loss.data[0])
			accs.append(acc.data[0])
			optimizer.zero_grad()
			loss.backward()
			if opt.max_norm > 0:
				torch.nn.utils.clip_grad_norm(model.parameters(), opt.max_norm)
			optimizer.step()
			idx_all += idx
			if float(idx_all) / opt.sent_size[0] - idx_prop > 0.2:
				idx_prop = float(idx_all) / opt.sent_size[0]
				logger.info("Training:\tEpoch : %d(batch %.2f)\tTime : %.4fs\tLoss : %.5f\tAccuracy : %.5f" \
					% (i, idx_prop, time.time() - start_time, loss.data[0], acc.data[0]))
		

		logger.info('Training:\tEpoch : %d\tTime : %.4fs\t Loss : %.5f\tAccuracy : %.5f ' \
			% (i, time.time() - start_time, np.mean(losses, axis=0), np.mean(accs, axis=0)))
		gc.collect()
		data_file.close()

		# evaluate
		model.eval()
		start_time = time.time()
		loss_val, p_val, r_val, f_val, a_val = evaluate(opt.valid_data_file)
		logger.info('Evaluation valid:\tEpoch : %d\tTime : %.4fs\tLoss : %.5f\tFscore : %.5f\tAccuracy: %.5f' % (i, time.time() - start_time, loss_val, f_val, a_val))
		start_time = time.time()
		loss_te, p_te, r_te, f_te, a_te = evaluate(opt.test_data_file, out_path=os.path.join(exp_path, 'test_iter'+str(i)+'_res'))
		logger.info('Evaluation test:\tEpoch : %d\tTime : %.4fs\tLoss : %.5f\tFscore : %.5f\tAccuracy: %.5f' % (i, time.time() - start_time, loss_te, f_te, a_te))

		if best_vloss > loss_val:
			model.save_model(os.path.join(exp_path, 'model_'+str(i)))
			best_vloss = loss_val
			logger.info('NEW BEST:\tEpoch : %d\tbest valid a : %.5f;\ttest a : %.5f' % (i, a_val, a_te))
			best_res['iter'] = i
			best_res['vp'], best_res['vr'], best_res['vf'], best_res['va'] = p_val, r_val, f_val, a_val
			best_res['tp'], best_res['tr'], best_res['tf'], best_res['ta'] = p_te, r_te, f_te, a_te
		elif best_ta < a_te:
			model.save_model(os.path.join(exp_path, 'model_'+str(i)))
	logger.info('BEST RESULT: \tEpoch : %d\tbest valid accuracy: %.5f\tbest test accuracy: %.5f' % (best_res['iter'], best_res['va'], best_res['ta']))

def evaluate(data_path, out_path=None):
	losses, accs = [], []
	TP, FP, FN, TN, T, F = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	data_file = open(data_path, "rb")
	if out_path is not None:
		out_file = open(out_path, 'wb')
		out_file.write('answer\tprediction\n')
	file_used = False
	while not file_used:
		# load a batch of data
		data_batch = []
		idx = 0
		while idx < opt.test_batchSize:
			data_line = data_file.readline()
			if data_line == '':
				file_used = True
				break
			elif data_line == '\n':
				idx += 1
			data_batch.append(data_line)
		if data_batch != []:
			input_batch = vocab.read_data_in_batch(data_batch, idx, use_char=opt.use_char, use_cuda=opt.cuda, evaluation=True)
		else:
			break
	
		# forward
		#probs, pred_ans = model(input_batch)
		#loss = loss_function(probs.contiguous(), input_batch['answer_idx'])
		loss, acc, pred_ans = model(input_batch)
		losses.append(loss.data[0])
		accs.append(acc.data[0])

		# p/r/f/a
		'''
		for idx, ans in enumerate(input_batch['answer_idx'].data):
			if ans == pred_ans.data[idx]:
				T += 1
			else:
				F += 1

		if T+F == 0:
			p, r, f, a = 0, 0, 0, 0
		else:
			#p, r, f = 100*TP/(TP+FP), 100*TP/(TP+FN), 100*2*TP/(2*TP+FN+FP)
			p, r, f = 0, 0, 0
			a = 100*T/(T+F)
		'''

		if out_path is not None:
			for i in xrange(len(input_batch['answer'].data)):
				answer = str(vocab.get_word(input_batch['answer'].data[i]))
				prediction = str(vocab.get_word(input_batch['cands'][i].data[pred_ans.data[i]]))
				#print answer, prediction
				out_file.write(answer+'\t'+prediction+'\n')
	data_file.close()
	if out_path is not None:
		out_file.close()
	loss = np.mean(losses, axis=0)
	a = np.mean(accs, axis=0)
	return loss, 0, 0, 0, a

if opt.testing:
	logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
	model.eval()
	start_time = time.time()
	loss_te, p_te, r_te, f_te, a_te = evaluate(opt.test_data_file, out_path=os.path.join(exp_path, 'test_res'))
	logger.info('Evaluation:\tTime : %.4fs\tLoss : %.5f\tFscore : %.5f\tAccuracy: %.5f' % (time.time() - start_time, loss_te, f_te, a_te))
else:
	logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
	model.train()
	train(opt.train_data_file)








