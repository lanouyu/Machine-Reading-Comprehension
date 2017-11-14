import torch
from torch.autograd import Variable
import operator
import json
import numpy as np

class Dictionary(object):
	def  __init__(self):
		self.word2idx = {}
		self.idx2word = []
		self.add_word('<pad>')
		self.add_word('<unk>')

	def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
		return self.word2idx[word]

	def get_word(self, idx):
		return self.idx2word[idx];

	def get_idx(self, word):
		if word not in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]

	def size(self):
		return len(self.idx2word)

	def save(self, path):
		with open(path, 'wb') as f:
			for idx in range(len(self.idx2word)):
				f.write(self.idx2word[idx]+' : '+str(idx)+'\n')

class Vocab(object):
	def __init__(self):
		self.dict = Dictionary()
		self.char_dict = Dictionary()

	def read_vocab_file(self, vocab_file, use_cuda=False, evaluation=False):
		flag_first = True
		word2vec = []
		with open(vocab_file, 'r') as f:
			for line in f:
				line = line.strip().split(' ')
				if flag_first:
					flag_first = False
					word_size = int(line[0])
					emb_dim = int(line[1])
					continue
				line_list = line
				word = line_list[0]
				for char in list(word):
					self.char_dict.add_word(char)
				vec = line_list[1:]
				self.dict.add_word(word)
				word2vec.append(vec)
		word2vec = torch.from_numpy(np.array(word2vec).astype(float))
		return word2vec, word_size, emb_dim, self.char_dict.size()

	def size():
		return self.dict.size()

	def get_word(self, idx):
		return self.dict.idx2word[idx];

	def get_idx(self, word):
		if word not in self.dict.word2idx:
			return self.dict.word2idx['<unk>']
		return self.dict.word2idx[word]

	def char_size():
		return self.char_dict.size()

	def char_get_word(self, idx):
		return self.char_dict.idx2word[idx];

	def char_get_idx(self, word):
		if word not in self.char_dict.word2idx:
			return self.char_dict.word2idx['<unk>']
		return self.char_dict.word2idx[word]

	def read_vocab_from_data(self, data_files):
		sent_size = []
		for data_file in data_files:
			with open(data_file, 'r') as f:
				sent_count = 0
				for line in f:
					line = line.strip()
					if line == '':
						sent_count += 1
						continue
					words = line.split()
					# ignore the first word, the number of lines
					if words[0] != '21':
						for word in words[1:]:
							self.dict.add_word(word)
					else:
						# cands must appear in context before
						for word in words[1:-1]:
							self.dict.add_word(word)
							for char in list(word):
								self.char_dict.add_word(char)
			sent_size.append(sent_count)
		return self.dict.size(), sent_size, self.char_dict.size()

	def save_vocab(self, vocab_path):
		self.dict.save(vocab_path)

	def read_data_in_batch(self, data_batch, batch_size, use_char=False, use_cuda=False, evaluation=False):
		data = {}
		data['context'] = [[]]
		data['context_mask'] = [[]]
		data['question'] = [[]]
		data['question_mask'] = [[]]
		#data['text'] = [[]]	# context + question
		#data['text_mask'] = [[]]
		data['cands'] = [[]] # 10 candidates
		data['cands_mask3'] = [[]] # candidate in doc (b, D, C)
		data['cands_mask2'] = [[]] # candidate in doc (b, D)
		data['answer'] = [[]] # 1 answer
		data['answer_idx'] = [[]] # the answer index in candidates
		data['cloze_pos'] = [[]] # the position of cloze in question
		data['feature_mask'] = [[]] # qe-comm, whether the word appears in question
		if use_char:
			data['token'] = [[]] # token in chars (#token, #max_char/token)
			data['token_mask'] = [[]]
			data['context_char'] = [[]] # token index in context
			data['question_char'] = [[]]
		token = [] # token, help for data['token']
		for i in xrange(batch_size - 1):
	 		for (k, v) in data.items():
				v.append([])

		# read data
		idx = 0
		for line in data_batch:
			line = line.strip()
			if line == '':
				idx += 1
				continue
			words = line.split()
			if words[0] != '21':
				for word in words[1:]:
					word_idx = self.dict.get_idx(word)
					data['context'][idx].append(word_idx)
					#data['text'][idx].append(word_idx)
					if use_char:
						if word not in token:
							token.append(word)
						data['context_char'][idx].append(token.index(word))
			else:
				# question
				for i, word in enumerate(words[1:-2]):
					word_idx = self.dict.get_idx(word)
					data['question'][idx].append(word_idx)
					#data['text'][idx].append(word_idx)
					if word == 'XXXXX':
						data['cloze_pos'][idx] = i
					if use_char:
						if word not in token:
							token.append(word)
						data['question_char'][idx].append(token.index(word))
				# candidates
				cands = words[-1].split('|')
				for i in xrange(10):
					if cands[i] == '':
						data['cands'][idx].append(self.dict.get_idx('|'))
					else:
						data['cands'][idx].append(self.dict.get_idx(cands[i]))
				# answer
				data['answer'][idx] = self.dict.get_idx(words[-2])
				data['answer_idx'][idx] = data['cands'][idx].index(data['answer'][idx])

		# transfer token to token_char
		if use_char:
			data['token'] = [[]]
			data['token_mask'] = [[]]
			max_word_len = max([len(line) for line in token])
			idx = 0
			for word in token:
				if idx > 0:
					data['token'].append([])
					data['token_mask'].append([])
				chars = list(word)
				for c in chars:
					data['token'][idx].append(self.char_dict.get_idx(c))
				pad_len = max_word_len - len(chars)
				data['token'][idx] += [self.char_dict.get_idx('<pad>')] * pad_len
				data['token_mask'][idx] = [1] * len(chars) + [0] * pad_len
				idx += 1
				

		# add pad
		max_context_len = max([len(line) for line in data['context']])
		max_question_len = max([len(line) for line in data['question']])
		#max_text_len = max([len(line) for line in data['text']])
		for i in xrange(batch_size):
			data_len = len(data['context'][i])
			pad_len = max_context_len - data_len
			for j, word in enumerate(data['context'][i]):
				# whether the word appears in question
				if word in data['question'][i]:
					data['feature_mask'][i] += [1]
				else:
					data['feature_mask'][i] += [0]
				# whether the word is a candidate and which
				data['cands_mask3'][i] += [[0] * 10]
				if word in data['cands'][i]:
					cand_idx = data['cands'][i].index(word)
					data['cands_mask3'][i][j][cand_idx] = 1
			data['feature_mask'][i] += [0] * pad_len
			data['cands_mask3'][i] += [[0] * 10] * pad_len
			data['context'][i] += [self.dict.get_idx('<pad>')] * pad_len
			data['context_char'][i] += [self.dict.get_idx('<pad>')] * pad_len
			data['context_mask'][i] = [1] * data_len + [0] * pad_len
		#for i in xrange(batch_size):
			data_len = len(data['question'][i])
			pad_len = max_question_len - data_len
			data['question'][i] += [self.dict.get_idx('<pad>')] * pad_len
			data['question_char'][i] += [self.dict.get_idx('<pad>')] * pad_len
			data['question_mask'][i] = [1] * data_len + [0] * pad_len
		'''
		for i in xrange(batch_size):
			data_len = len(data['text'][i])
			pad_len = max_text_len - data_len
			data['text'][i] += [self.dict.get_idx('<pad>')] * pad_len
			data['text_mask'][i] = [1] * data_len + [0] * pad_len
		'''
		data['cands_mask2'] = np.array(np.sum(data['cands_mask3'], axis=2)>0, dtype=int)

		# convert to Variable
		try:
			for (k, v) in data.items():
				data[k] = Variable(torch.LongTensor(data[k]), volatile=evaluation)
				if use_cuda:
					data[k] = data[k].cuda()
		except:
			print k, v

		return data

	def read_data_from_file(self, data_file, use_cuda=False, evaluation=False):
		data = {}
		data['context'] = [[]]
		data['context_mask'] = [[]]
		data['question'] = [[]]
		data['question_mask'] = [[]]
		data['text'] = [[]]	# context + question
		data['text_mask'] = [[]]
		data['cands'] = [[]] # 10 candidates
		data['cands_mask3'] = [[]] # candidate in doc (b, D, C)
		data['answer'] = [[]] # 1 answer
		data['answer_idx'] = [[]] # the answer index in candidates
		data['cloze_pos'] = [[]] # the position of cloze in question
		data['feature_mask'] = [[]] # qe-comm, whether the word appears in question			

		# read data
		with open(data_file, 'r') as f:
			idx = 0
			for line in f:
				line = line.strip()
				if line == '':
					idx += 1
					continue
				words = line.split()
				if words[0] == '1' and idx > 0:
					for (k, v) in data.items():
						v.append([])
				if words[0] != '21':
					for word in words[1:]:
						word_idx = self.dict.get_idx(word)
						data['context'][idx].append(word_idx)
						data['text'][idx].append(word_idx)
				else:
					# question
					for i, word in enumerate(words[1:-2]):
						word_idx = self.dict.get_idx(word)
						data['question'][idx].append(word_idx)
						data['text'][idx].append(word_idx)
						if word == 'XXXXX':
							data['cloze_pos'][idx] = i
					# candidates
					cands = words[-1].split('|')
					for i in xrange(10):
						if cands[i] == '':
							data['cands'][idx].append(self.dict.get_idx('|'))
						else:
							data['cands'][idx].append(self.dict.get_idx(cands[i]))
					# answer
					data['answer'][idx] = self.dict.get_idx(words[-2])
					data['answer_idx'][idx] = data['cands'][idx].index(data['answer'][idx])

		# add pad
		batch_size = len(data['answer'])
		max_context_len = max([len(line) for line in data['context']])
		max_question_len = max([len(line) for line in data['question']])
		max_text_len = max([len(line) for line in data['text']])
		for i in xrange(batch_size):
			data_len = len(data['context'][i])
			pad_len = max_context_len - data_len
			for j, word in enumerate(data['context'][i]):
				# whether the word appears in question
				if word in data['question'][i]:
					data['feature_mask'][i] += [1]
				else:
					data['feature_mask'][i] += [0]
				# whether the word is a candidate and which
				data['cands_mask3'][i] += [[0] * 10]
				if word in data['cands'][i]:
					cand_idx = data['cands'][i].index(word)
					data['cands_mask3'][i][j][cand_idx] = 1
			data['feature_mask'][i] += [0] * pad_len
			data['cands_mask3'][i] += [[0] * 10] * pad_len
			data['context'][i] += [self.dict.get_idx('<pad>')] * pad_len
			data['context_mask'][i] = [1] * data_len + [0] * pad_len
		#for i in xrange(batch_size):
			data_len = len(data['question'][i])
			pad_len = max_question_len - data_len
			data['question'][i] += [self.dict.get_idx('<pad>')] * pad_len
			data['question_mask'][i] = [1] * data_len + [0] * pad_len
		'''
		for i in xrange(batch_size):
			data_len = len(data['text'][i])
			pad_len = max_text_len - data_len
			data['text'][i] += [self.dict.get_idx('<pad>')] * pad_len
			data['text_mask'][i] = [1] * data_len + [0] * pad_len
		'''

		# convert to Variable
		try:
			for (k, v) in data.items():
				data[k] = Variable(torch.LongTensor(data[k]), volatile=evaluation)
				if use_cuda:
					data[k] = data[k].cuda()
		except:
			print k, v
		return data









