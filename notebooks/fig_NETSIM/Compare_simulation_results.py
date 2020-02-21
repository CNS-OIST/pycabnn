# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import data_io as io

from pathlib import Path

# %%
# boutput_path = Path('/Users/shhong/Dropbox/network_data/output.5028956/') #BREP
# poutput_path = Path('/Users/shhong/Dropbox/network_data/output.5030124/') #PyBREP
boutput_path = Path('../../test_data/simulation_data/BREP/') #BREP
poutput_path = Path('../../test_data/simulation_data/Pycabnn/') #Pycabnn

bout = io.OutputReader(boutput_path)
pout = io.OutputReader(poutput_path)

df_b = bout.read_connectivity('aa', 'goc')
df_p = pout.read_connectivity('aa', 'goc')

gocp = pout.read_spike_data('goc')
gocb = bout.read_spike_data('goc')
# srcs = np.load(output_path / 'Axon.npy')
# tgts = np.load(output_path / 'AAtoGoCtargets.npy')
# grcxy = np.loadtxt(output_path / 'GCcoordinates.sorted.dat')
# gocxy = np.loadtxt(output_path / 'GoCcoordinates.sorted.dat')

# df = dd.from_array(np.vstack((srcs, tgts)).T, columns=('src', 'tgt'))

# %%
_, axs = plt.subplots(figsize=(8.5/2.54, 8.5/2.54*5/8), nrows=2, sharex=True)
axs[0].plot(gocb['time'], gocb['x']/1e3, '|', c='grey', markersize=1)
axs[1].plot(gocp['time'], gocp['x']/1e3, '|k', markersize=1)
axs[0].set(yticks=[0, 1.5], ylabel='x (mm)')
axs[1].set(xlim=[350, 650], xlabel='time (ms)', yticks=[0, 1.5], ylabel='x (mm)')
plt.tight_layout()
plt.savefig('goc_x.jpg', dpi=600)
plt.savefig('goc_x.pdf', dpi=600)

# %%
_, axs = plt.subplots(figsize=(8.5/2.54, 8.5/2.54*5/8), nrows=2, sharex=True)
axs[0].plot(gocb['time'], gocb['y']/1e3, '|', c='grey', markersize=1)
axs[1].plot(gocp['time'], gocp['y']/1e3, '|k', markersize=1)
axs[0].set(yticks=[0, 0.7], ylabel='y (mm)')
axs[1].set(xlim=[350, 650], xlabel='time (ms)', yticks=[0, 0.7], ylabel='y (mm)')
plt.tight_layout()
plt.savefig('goc_y.jpg', dpi=600)
plt.savefig('goc_y.pdf', dpi=600)

# %%
bcons_per_goc = df_b.groupby('cell').count()
pcons_per_goc = df_p.groupby('cell').count()

_, axs = plt.subplots(ncols=2, figsize=(8.5/2.54*2, 8.5/2.54), sharey=True)
_ = axs[0].hist(bcons_per_goc['pre'], 30, color='grey')
_ = axs[1].hist(pcons_per_goc['pre'], 30, color='k')
axs[0].set(xlabel='AA synapses per GoC', ylabel='count')
axs[1].set(xlabel='AA synapses per GoC')

plt.tight_layout()
plt.savefig('aa_goc.jpg', dpi=600)
plt.savefig('aa_goc.pdf', dpi=600)

# %%
df_b = bout.read_connectivity('pf', 'goc')
df_p = pout.read_connectivity('pf', 'goc')

bcons_per_goc = df_b.groupby('cell').count()
pcons_per_goc = df_p.groupby('cell').count()

_, axs = plt.subplots(ncols=2, figsize=(8.5/2.54*2, 8.5/2.54), sharey=True)
_ = axs[0].hist(bcons_per_goc['pre'], 30, color='grey')
_ = axs[1].hist(pcons_per_goc['pre'], 30, color='k')
axs[0].set(xlabel='PF synapses per GoC', ylabel='count')
axs[1].set(xlabel='PF synapses per GoC')

plt.tight_layout()
plt.savefig('pf_goc.jpg', dpi=600)
plt.savefig('pf_goc.pdf', dpi=600)

