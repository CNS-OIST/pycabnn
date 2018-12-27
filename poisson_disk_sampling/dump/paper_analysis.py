{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:\\Users\\idsan\\OneDrive\\문서\\GitHub\\pybrep\\poisson_disk_sampling\n"
     ]
    }
   ],
   "source": [
    "cd C:\\Users\\idsan\\OneDrive\\문서\\GitHub\\pybrep\\poisson_disk_sampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy import spatial\n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "real_point = np.loadtxt('cells_GINIX_GL.dat', skiprows=1)\n",
    "conversion_factor = 2*0.178\n",
    "real_point = real_point * conversion_factor\n",
    "def filter_real(real_point):\n",
    "    ii = np.logical_and(real_point[:, 0] > 30, real_point[:, 0] < 300 - 30)\n",
    "    real_point = real_point[ii, 0:3]\n",
    "    ii = np.logical_and(real_point[:, 1] > 140 + 30, real_point[:, 1] < 300 - 30)\n",
    "    real_point = real_point[ii, 0:3]\n",
    "    ii = np.logical_and(real_point[:, 2] > 30, real_point[:, 2] < 300 - 30)\n",
    "    real_point = real_point[ii, 0:3]\n",
    "    return real_point\n",
    "\n",
    "real_point = filter_real(real_point)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "poisson_point = np.loadtxt('Multipy point 3.txt')\n",
    "poisson_point[:, 0] = poisson_point[:, 0] * 0.3\n",
    "poisson_point[:, 1] = poisson_point[:, 1]* 0.3\n",
    "poisson_point[:, 2] = poisson_point[:, 2] * 0.3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "rand_point = np.random.rand(40000, 3)\n",
    "rand_point[:, 0] = rand_point[:, 0] * 300\n",
    "rand_point[:, 1] = rand_point[:, 1] * 150 + 150\n",
    "rand_point[:, 2] = rand_point[:, 2] * 300"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "tree_real = spatial.cKDTree(real_point[:, 0:3])\n",
    "tree_poisson = spatial.cKDTree(poisson_point)\n",
    "tree_rand = spatial.cKDTree(rand_point)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "d_real, _ = tree_real.query(real_point[:, 0:3], k=2)\n",
    "d_poisson, _ = tree_poisson.query(poisson_point, k=2)\n",
    "d_rand, _ = tree_rand.query(rand_point, k=2)\n",
    "\n",
    "d_real = d_real[:, 1]\n",
    "d_poisson = d_poisson[:, 1]\n",
    "d_rand = d_rand[:, 1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "def count_neighborhood(x, tree, r):\n",
    "    temp = tree.query_ball_point(x[:, 0:3], r)\n",
    "    count = np.array([len(x)-1 for x in temp])\n",
    "    return count\n",
    "def count_neighborhood1(x, tree, r):\n",
    "    temp = tree.query_ball_point(x, r)\n",
    "    count = np.array([len(x)-1 for x in temp])\n",
    "    return count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "mean_count_real = []\n",
    "mean_count_poisson = []\n",
    "mean_count_rand = []\n",
    "var_count_real = []\n",
    "var_count_poisson = []\n",
    "var_count_rand = []\n",
    "rs = np.arange(0.1,30,0.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for r in tqdm(rs):\n",
    "    c0 = count_neighborhood(real_point, tree_real, r)\n",
    "    mean_count_real.append(np.mean(c0))\n",
    "    var_count_real.append(np.var(c0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|▎                                                                             | 1/299 [03:19<16:31:24, 199.61s/it]"
     ]
    }
   ],
   "source": [
    "for r in tqdm(rs):\n",
    "    c1 = count_neighborhood1(poisson_point, tree_poisson, r)\n",
    "    mean_count_poisson.append(np.mean(c1))\n",
    "    var_count_poisson.append(np.var(c1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for r in tqdm(rs):\n",
    "    c2 = count_neighborhood1(rand_point, tree_rand, r)\n",
    "    mean_count_rand.append(np.mean(c2))\n",
    "    var_count_rand.append(np.var(c2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mean_count_real = np.array(mean_count_real)\n",
    "mean_count_poisson = np.array(mean_count_poisson)\n",
    "mean_count_rand = np.array(mean_count_rand)\n",
    "\n",
    "var_count_real = np.array(var_count_real)\n",
    "var_count_poisson = np.array(var_count_poisson)\n",
    "var_count_rand = np.array(var_count_rand)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure()\n",
    "\n",
    "    #subplot setting\n",
    "ax1 = fig.add_subplot(2, 2, 1)\n",
    "ax2 = fig.add_subplot(2, 2, 2)\n",
    "ax3 = fig.add_subplot(2, 2, 3)\n",
    "ax4 = fig.add_subplot(2, 2, 4)\n",
    "\n",
    "ax1.set_xlabel('radius', fontsize = 15)\n",
    "ax1.set_ylabel('Mean', fontsize = 15)\n",
    "ax2.set_xlabel('radius', fontsize = 15)\n",
    "ax2.set_ylabel('Standard Deviation', fontsize = 15)\n",
    "ax3.set_xlabel('radius', fontsize = 15)\n",
    "ax3.set_ylabel('Density Fluctuation', fontsize = 15)\n",
    "ax4.set_xlabel('Mean', fontsize = 15)\n",
    "ax4.set_ylabel('Standard Deviation', fontsize = 15)\n",
    "\n",
    "ax1.set_title('Mean')\n",
    "ax2.set_title('Standard Devication')\n",
    "ax3.set_title('Density Fluctuation')\n",
    "ax4.set_title('Mean-SD')\n",
    "\n",
    "ax1.plot(rs, mean_count_real/rs**3, rs, mean_count_poisson/rs**3, rs, mean_count_rand/rs**3)\n",
    "ax2.plot(rs, np.sqrt(var_count_real)/rs**3, rs, np.sqrt(var_count_poisson)/rs**3, rs, np.sqrt(var_count_rand)/rs**3)\n",
    "ax3.plot(rs, np.sqrt(var_count_real)/rs**6, rs, np.sqrt(var_count_poisson)/rs**6, rs, np.sqrt(var_count_rand)/rs**6,  )\n",
    "ax4.plot(mean_count_real, np.sqrt(var_count_real), mean_count_poisson, np.sqrt(var_count_poisson), mean_count_rand, np.sqrt(var_count_rand))\n",
    "\n",
    "plt.close('all')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
