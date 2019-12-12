import numpy as np

runtimes = ['5m33.717s',
            '5m29.660s',
            '5m40.448s',
            '5m31.794s',
            '5m30.940s']

runtimes = [x.replace('s', '') for x in runtimes]
runtimes = [x.replace('m', '*60+') for x in runtimes]
runtimes = [eval(x) for x in runtimes]

print(np.mean(runtimes), np.std(runtimes))