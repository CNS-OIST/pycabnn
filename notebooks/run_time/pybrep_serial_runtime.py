import numpy as np

runtimes = ['791.78s',
            '819.38s',
            '899.83s',
            '827.53s',
            '807.54s']

runtimes = [x.replace('s', '') for x in runtimes]
runtimes = [float(x) for x in runtimes]
print(np.mean(runtimes), np.std(runtimes))