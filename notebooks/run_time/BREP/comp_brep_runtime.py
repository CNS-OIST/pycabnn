import os
import re
import numpy as np

log_list = os.listdir('.')
log_list = [x for x in log_list if x.endswith('.eml')]

re1='.*?'	# Non-greedy match on filler
re2='\\d+'	# Uninteresting: int
re3='.*?'	# Non-greedy match on filler
re4='\\d+'	# Uninteresting: int
re5='.*?'	# Non-greedy match on filler
re6='(\\d+)'	# Integer Number 1
re7='.*?'	# Non-greedy match on filler
re8='(\\d+)'	# Integer Number 2

rg = re.compile(re1+re2+re3+re4+re5+re6+re7+re8,re.IGNORECASE|re.DOTALL)

secs = []
for i, txt in enumerate(log_list):
    m = rg.search(txt)
    int2=int(m.group(1))
    int3=int(m.group(2))
    print(i+1, txt, int2, int3)
    secs.append(int2*60+int3)

# print(secs)
print('Trials = {}, t = {} +- {} secs'.format(len(secs), np.mean(secs), np.std(secs)))