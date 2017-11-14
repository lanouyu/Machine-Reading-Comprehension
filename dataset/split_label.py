
import sys,random
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-s", "--seed", action="store",
                  dest="seed",
                  default=1000,
                  help="random seed")
parser.add_option("--prop", action="store",
                  dest="prop",
                  default="4:1",
                  help="the proptation of train to valid, default 4:1")
(options, args) = parser.parse_args()
seed = int(options.seed)
train_num = int(options.prop.split(':')[0])
valid_num = int(options.prop.split(':')[1])

if len(args) != 1:
  parser.usage = "%prog [options] <data-file>"
  parser.print_help()
  sys.exit(0)

data_file = open(args[0],'rb')
line_list = []
line_single = []
for line in data_file:
  word = line.strip('\n')
  if (word == ''):
  	line_list.append(line_single)
	line_single = []
  else:
  	line_single.append(word)
  
length = len(line_list)
random.seed(seed)

idx_list = range(length)
#random.shuffle(idx_list)

train_file = open('valid.txt', 'wb')
valid_file = open('test.txt', 'wb')
	
i = 0
for idx in idx_list:
  if i < length * train_num / (train_num + valid_num):
    for word in line_list[idx]:
  	  train_file.write(word + '\n')
    train_file.write('\n')
  else:
    for word in line_list[idx]:
  	  valid_file.write(word + '\n')
    valid_file.write('\n')
  i += 1

data_file.close()
train_file.close()
valid_file.close()
