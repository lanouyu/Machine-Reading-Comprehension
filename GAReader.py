import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class GAReader(nn.Module):
	def __init__(self, num_layers, vocab, vocab_size, char_size=0, dropout=0.5, gru_dim=100, 
		word2vec=None, init_word_size=0,fix_emb=False, embed_dim=100, char_emb_dim=100, char_gru_dim=50,
		use_cuda=False, use_feature=False, use_char=False, ga_fn='mul'):
		super(GAReader, self).__init__()
		self.num_layers = num_layers
		self.vocab = vocab
		self.vocab_size = vocab_size
		self.dropout = dropout
		self.gru_dim = gru_dim
		self.embed_dim = embed_dim
		self.char_emb_dim = char_emb_dim
		self.char_gru_dim = char_gru_dim
		self.use_cuda = use_cuda
		self.word2vec = word2vec
		self.fix_emb = fix_emb
		self.use_char = use_char
		self.use_feature = use_feature
		self.ga_fn = ga_fn
		self.char_size = char_size

		# dropout
		self.dropout_layer = nn.Dropout(self.dropout)
		
		# word embedding
		self.embedding = nn.Embedding(vocab_size, self.embed_dim)
		if self.word2vec is not None:
			self.embedding.weight.data[:init_word_size].copy_(word2vec)
		if self.fix_emb:
			self.embedding.weight.requires_grad = False

		self.gru_input_size = self.embed_dim
		if self.num_layers == 1:
			self.final_input_size = self.embed_dim
		else:
			self.final_input_size = self.gru_dim * 2
		
		# char-level embedding
		if self.use_char:
			self.char_embed = nn.Embedding(self.char_size, self.char_emb_dim)
			self.char_gru = nn.GRU(
				input_size=self.char_emb_dim,
				hidden_size=self.char_gru_dim,
				dropout=self.dropout,
				batch_first=True,
				bidirectional=True)
			self.char_fw = nn.Linear(self.char_gru_dim, self.embed_dim // 2)
			self.char_bk = nn.Linear(self.char_gru_dim, self.embed_dim // 2)
			self.gru_input_size += self.embed_dim // 2
		
		# qe-comm feature embedding, whether appear in question
		if self.use_feature:
			self.feat_embed = nn.Embedding(2, 2)
			self.final_input_size += 2

		# intermidate GRU layers
		self.inter_d_layers = nn.ModuleList()
		self.inter_q_layers = nn.ModuleList()
		for i in xrange(self.num_layers-1):
			d_layer = nn.GRU(
				input_size=self.gru_input_size if i==0 else 2 * self.gru_dim,
				hidden_size=self.gru_dim,
				batch_first=True,
				bidirectional=True)
			q_layer = nn.GRU(
				input_size=self.gru_input_size,
				hidden_size=self.gru_dim,
				batch_first=True,
				bidirectional=True)
			self.inter_d_layers.append(d_layer)
			self.inter_q_layers.append(q_layer)

		# final GRU layer
		self.final_d_layer = nn.GRU(
			input_size=self.final_input_size,
			hidden_size=self.gru_dim,
			batch_first=True,
			bidirectional=True)
		self.final_q_layer = nn.GRU(
			input_size=self.gru_input_size,
			hidden_size=self.gru_dim,
			batch_first=True,
			bidirectional=True)

	def gru_layer(self, inputs, mask, net_layer):
		# inputs (b, D, dim)
		# mask (b, D, dim)
		# sort
		seq_lengths = torch.sum(mask, dim=-1).squeeze(-1) # (b)
		sorted_len, sorted_idx = seq_lengths.sort(0, descending=True) # sort
		index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(inputs) # (b, D, dim), elems in each batch are same, sorted_idx
		sorted_inputs = inputs.gather(0, index_sorted_idx.long()) # (b, D, dim)
		# run
		packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
			sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
		packed_out, _ = net_layer(packed_seq)
		unpacked_out, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(
			packed_out, batch_first=True)
		# unsort
		_, index_idx = sorted_idx.sort(0, descending=False)
		idx = index_idx.view(-1, 1, 1).expand_as(unpacked_out)
		seq_out = unpacked_out.gather(0, idx.long())
		return seq_out, seq_lengths

	def gated_attention(self, d, q, q_mask, ga_fn='mul'):
		# d (b, D, gru_dim * direct)
		# q (b, Q, gru_dim * direct)
		inter = torch.bmm(d, q.permute(0, 2, 1))
		a = F.softmax(inter.view(-1, inter.size(-1))).view_as(inter) * \
			q_mask.unsqueeze(1).float().expand_as(inter)
		a = a / torch.sum(a, dim=2).expand_as(a)
		#a = F.softmax(torch.bmm(d, q.permute(0, 2, 1))) # (b, D, Q)
		qt = torch.bmm(a, q) # (b, D, gru_dim)
		if ga_fn == 'mul':
			x = torch.mul(d, qt)
		elif ga_fn == 'add':
			x = d + qt
		else:
			x = torch.cat([d, qt], axis=2)
		return x

	def pred_layer(self, d, q, data):
		cloze_pos = data['cloze_pos'].view(-1, 1).expand(q.size(0), q.size(2)).unsqueeze(1) # (b, 1, dim)
		q = q.gather(1, cloze_pos.long()).squeeze(1) # (b, dim), the dim of XXXXX
		p = torch.squeeze(torch.bmm(d, q.unsqueeze(dim=-1)), dim=-1) # (b, D)
		# if self.use_feature:
		# 	pm = F.softmax(p * data['feature_mask'].float()).unsqueeze(1) # (b, 1, D)
		# else:
		# 	pm = F.softmax(p).unsqueeze(1) # (b, 1, D)
		pm = F.softmax(p * data['cands_mask2'].float()).unsqueeze(1) # (b, 1, D)
		probs = torch.squeeze(torch.bmm(pm, data['cands_mask3'].float()), dim=1) # (b, C)
		#probs = F.log_softmax(probs)
		return probs

	def forward(self, data):
		d_embed = self.embedding(data['context'])
		q_embed = self.embedding(data['question'])
		
		if self.use_char:
			c_embed = self.char_embed(data['token']) # (T, W, dim)
			c_gru_out, c_out_len = self.gru_layer(c_embed, data['token_mask'], self.char_gru) # (T, W, dim*2), (T)
			out_last_idx = (c_out_len - 1).view(-1, 1).expand(c_gru_out.size(0), c_gru_out.size(2)).unsqueeze(1)
			out_last = c_gru_out.gather(1, out_last_idx.long()).squeeze() # (T, dim*2)
			c_fw_out = self.char_fw(out_last[:, :self.char_gru_dim])
			c_bk_out = self.char_bk(out_last[:, self.char_gru_dim:])
			c_out = c_fw_out + c_bk_out # (T, emb_dim//2)
			d_c_embed = c_out.index_select(0, data['context_char'].view([-1])).view(
				list(data['context_char'].data.size()) + [self.embed_dim // 2]) # (b, D, emb_dim//2)
			q_c_embed = c_out.index_select(0, data['question_char'].view([-1])).view(
				list(data['question_char'].data.size()) + [self.embed_dim // 2]) # (b, Q, emb_dim//2)
			d_embed = torch.cat([d_embed, d_c_embed], dim=-1) # (b, D, emb_dim*1.5)
			q_embed = torch.cat([q_embed, q_c_embed], dim=-1) # (b, Q, emb_dim*1.5)
		
		# intermidate layers
		#drop_d_embed = self.dropout_layer(d_embed)
		#drop_q_embed = self.dropout_layer(q_embed)
		drop_d_embed = d_embed
		drop_q_embed = q_embed

		for i in xrange(self.num_layers-1):
			d_gru, _ = self.gru_layer(drop_d_embed, data['context_mask'], self.inter_d_layers[i])
			q_gru, _ = self.gru_layer(drop_q_embed, data['question_mask'], self.inter_q_layers[i])
			ga_out = self.gated_attention(d_gru, q_gru, data['question_mask'], ga_fn=self.ga_fn)
			drop_d_embed = self.dropout_layer(ga_out)

		# final layer
		if self.use_feature:
			f_embed = self.feat_embed(data['feature_mask'])
			d_embed = torch.cat([drop_d_embed, f_embed], dim=-1)
			drop_d_embed = d_embed
		d_gru, _ = self.gru_layer(drop_d_embed, data['context_mask'], self.final_d_layer)
		q_gru, _ = self.gru_layer(drop_q_embed, data['question_mask'], self.final_q_layer)
		probs = self.pred_layer(d_gru, q_gru, data)
		loss = torch.mean(-torch.log(probs.gather(1, data['answer_idx'].unsqueeze(1))))
		_, pred_ans = torch.max(probs, dim=1)
		pred_ans = pred_ans.squeeze(1)
		acc = torch.mean(torch.eq(pred_ans, data['answer_idx']).float())
		#print 'pred_ans', pred_ans
		#return probs, pred_ans
		return loss, acc, pred_ans


	def load_model(self, load_dir):
		self.load_state_dict(torch.load(open(load_dir)))

	def save_model(self, save_dir):
		torch.save(self.state_dict(), open(save_dir, 'wb'))
















		

