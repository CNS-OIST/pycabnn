#!/usr/bin/env python
"""Unpack the connectivity saved in *.db or *.h5 into text files

Usage:
    unpack_db.py <database> <nblocks> <ntargets>
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
import pycabnn
import ipyparallel as ipp

ipp.register_joblib_backend()


def main(db, nblocks, ntargets):
    # c = conn.cursor()
    # c.execute('SELECT MAX(target) FROM connection')
    # ntargets, = c.fetchone()
    # print(ntargets)

    prefix = Path(db).resolve()
    prefix = str(prefix.parent / prefix.stem)

    def sqlite3_ffetch(k):
        with sqlite3.connect(db) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM connection WHERE target={}".format(k), conn
            )
        return df

    def hdf5_ffetch(k):
        return pd.read_hdf(db, "connection", where="target={}".format(k))

    dbkind = Path(db).suffix
    if dbkind == ".db":
        ffetch = sqlite3_ffetch
    elif dbkind == ".h5":
        ffetch = hdf5_ffetch

    def run_block(i):
        m = np.arange(i, ntargets, nblocks)

        df = pd.concat(
            Parallel(n_jobs=-1)(delayed(ffetch)(k) for k in m), ignore_index=True
        )

        fn_src = prefix + ("sources" + str(i) + ".dat")
        fn_tar = prefix + ("targets" + str(i) + ".dat")
        fn_segs = prefix + ("segments" + str(i) + ".dat")
        fn_dis = prefix + ("distances" + str(i) + ".dat")

        with open(fn_tar, "w") as f_tar, open(fn_src, "w") as f_src, open(
            fn_segs, "w"
        ) as f_segs, open(fn_dis, "w") as f_dis:
            np.savetxt(f_src, df["source"].values, delimiter=" ", fmt="%d")
            np.savetxt(f_tar, df["target"].values, delimiter=" ", fmt="%d")

            z = np.vstack([df["segment"].values, df["branch"].values])
            np.savetxt(f_segs, z.T, delimiter=" ", fmt="%d")

            np.savetxt(f_dis, df["distance"].values, delimiter=" ")

    for i in trange(nblocks):
        run_block(i)


if __name__ == "__main__":
    args = docopt(__doc__, version=pycabnn.__version__)
    main(args["<database>"], int(args["<nblocks>"]), int(args["<ntargets>"]))
