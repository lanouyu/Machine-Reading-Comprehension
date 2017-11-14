

import sys,random
from optparse import OptionParser

parser = OptionParser()
(options, args) = parser.parse_args()

if len(args) != 1:
  parser.usage = "%prog [options] <data-file>"
  parser.print_help()
  sys.exit(0)

data_file = open(args[0],'rb')

num = 0
for line in data_file:
	if (line == '\n'):
		num += 1
  
print num

data_file.close()
