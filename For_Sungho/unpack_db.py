#!/usr/bin/env python
"""Unpack the connectivity saved in *.db into text files

Usage:
    unpack_db.py <sqlite_database> <nblocks> <ntargets>
    unpack_db.py -h | --help

Options:
    -h --help   Show this screen.

"""

from docopt import docopt
import numpy as np
import pandas as pd
import sqlite3
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import trange


def main(db, nblocks, ntargets):
    # c = conn.cursor()
    # c.execute('SELECT MAX(target) FROM connection')
    # ntargets, = c.fetchone()
    # print(ntargets)

    prefix = Path(db).resolve()
    prefix = str(prefix.parent / prefix.stem)

    def run_block(i):
        m = np.arange(i, ntargets, nblocks)

        def ffetch(k):
            with sqlite3.connect(db) as conn:
                df = pd.read_sql_query(
                        'SELECT * FROM connection WHERE target={}'.format(k),
                        conn
                    )
            return df

        df = pd.concat(
                Parallel(n_jobs=-1)(delayed(ffetch)(k) for k in m),
                ignore_index=True
            )

        fn_src =  prefix + ('source' + str(i) + '.dat')
        fn_tar =  prefix + ('target' + str(i) + '.dat')
        fn_segs = prefix + ('segments' + str(i) + '.dat')
        fn_dis  = prefix + ('distances' + str(i) + '.dat')

        with open(fn_tar, 'w') as f_tar, open(fn_src, 'w') as f_src, open(fn_segs, 'w') as f_segs, open(fn_dis, 'w') as f_dis:
            np.savetxt(f_src, df['source'].values, delimiter=' ', fmt='%d')
            np.savetxt(f_tar, df['target'].values, delimiter=' ', fmt='%d')

            z = np.vstack([df['segment'].values, df['branch'].values])
            np.savetxt(f_segs, z.T, delimiter=' ', fmt='%d')

            np.savetxt(f_dis, df['distance'].values, delimiter=' ')

    for i in trange(nblocks):
        run_block(i)


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args['<sqlite_database>'], int(args['<nblocks>']), int(args['<ntargets>']))