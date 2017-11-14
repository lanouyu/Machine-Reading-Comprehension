
net=GAReader
did=-1
mb=32 # batch size
tmb=10 # test batch size
lr=0.0005
dropout=0.4
nl=3 # num_layers
tf='train.txt'
gd=128 # gru_dim
ie='../data/word2vec_glove.txt' # init embedding file
python main.py --data_path ../data --net $net --deviceId $did --batchSize $mb --lr $lr \
	--dropout $dropout --num_layers $nl --test_batchSize $tmb --train_file $tf --gru_dim $gd \
	--init_emb $ie --fix_emb --use_feature --use_char
