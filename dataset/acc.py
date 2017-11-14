import sys
import numpy as np 

answer = open('answer', 'rb')
ans_exp = open(sys.argv[1], 'rb')

ans = answer.readlines()
pred = ans_exp.readlines()

eq = 0.0
for i in xrange(len(ans)):
	if ans[i] == pred[i]:
		eq += 1
	else:
		print i, ans[i].strip(), pred[i].strip()

print eq / len(ans)
