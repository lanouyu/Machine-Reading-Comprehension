import sys

answer = open('answer', 'wb')
ans_exp = open('answer_nlp', 'wb')
order_file = open('order.txt', 'rb')
test_file = open('test_iter5_res','rb')

order = [int(x.strip()) for x in order_file]
test = [(x.strip()).split('\t') for x in test_file]
print order
print test
for i in order:
	answer.write(test[i][0]+'\n')
	ans_exp.write(test[i][1]+'\n')