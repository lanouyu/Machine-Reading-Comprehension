
import sys,random
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-s", "--seed", action="store",
                  dest="seed",
                  default=1000,
                  help="random seed")
(options, args) = parser.parse_args()
seed = int(options.seed)

if len(args) != 1:
  parser.usage = "%prog [options] <test-file>"
  parser.print_help()
  sys.exit(0)

o_train = open('../CBTest/data/cbtest_NE_train.txt', 'rb')
o_valid = open('../CBTest/data/cbtest_NE_valid_2000ex.txt', 'rb')
o_test = open('../CBTest/data/cbtest_NE_test_2500ex.txt', 'rb')
t_train = open('train.txt', 'rb')
t_valid = open('dev.txt', 'rb')
t_test = open(args[0], 'wb')

key_list = []
for line in t_train:
  word = line.strip('\n').split(' ')
  if word[0] == '21':
  	key_list.append(line)
print('t_train completed')

for line in t_valid:
  word = line.strip('\n').split(' ')
  if word[0] == '21':
    key_list.append(line)
print('t_valid completed')

lines = ''	
for line in o_train:
  word = line.strip('\n').split(' ')
  lines += line
  if word[0] == '21':
    if line not in key_list:
      t_test.write(lines + '\n')
      print('train not in file!')
    else:
      print('find')
    lines = ''
print('o_train completed')
for line in o_valid:
  word = line.strip('\n').split(' ')
  lines += line
  if word[0] == '21':
    if line not in key_list:
      t_test.write(lines + '\n')
      print('valid not in file!')
    lines = ''
print('o_valid completed')
for line in o_test:
  word = line.strip('\n').split(' ')
  lines += line
  if word[0] == '21':
    if line not in key_list:
      t_test.write(lines)
    lines = ''
print('o_test completed')

o_train.close()
o_valid.close()
o_test.close()
t_train.close()
t_valid.close()
t_test.close()
