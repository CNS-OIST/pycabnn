{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "s = np.random.seed(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'MF_coordinates' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-5-28b6af12496a>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mMFdensity\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m1650\u001b[0m \u001b[1;31m# 5000;%cells/mm2%190\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mquit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 6\u001b[1;33m \u001b[1;32mdel\u001b[0m \u001b[0mMF_coordinates\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'MF_coordinates' is not defined"
     ]
    }
   ],
   "source": [
    "quit()\n",
    "Longaxis = 1500 #185%eval (readParameters ('GoCxrange', 'Parameters.hoc'))  % um\n",
    "Shortaxis = 700 #185%eval (readParameters ('GoCxrange', 'Parameters.hoc'))  % um\n",
    "MFdensity = 1650 # 5000;%cells/mm2%190\n",
    "quit()\n",
    "del MF_coordinates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'Longaxis' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-7-351cadd1c11a>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mXinstantiate\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m64\u001b[0m\u001b[1;33m+\u001b[0m\u001b[1;36m40\u001b[0m \u001b[1;31m#297+40\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mYinstantiate\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m84\u001b[0m\u001b[1;33m+\u001b[0m\u001b[1;36m40\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mbox_fac\u001b[0m \u001b[1;31m#474+40*box_fac\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m \u001b[0mnumMF\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mLongaxis\u001b[0m\u001b[1;33m+\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mXinstantiate\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m*\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mShortaxis\u001b[0m\u001b[1;33m+\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mYinstantiate\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mMFdensity\u001b[0m\u001b[1;33m*\u001b[0m\u001b[1;36m1e-6\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'Longaxis' is not defined"
     ]
    }
   ],
   "source": [
    "fid = np.load('datasp.data', 'w+')\n",
    "box_fac = 2.5\n",
    "Xinstantiate = 64+40 #297+40\n",
    "Yinstantiate = 84+40*box_fac #474+40*box_fac\n",
    "numMF = (Longaxis+(2*Xinstantiate))*(Shortaxis+(2*Yinstantiate))*MFdensity*1e-6\n",
    "plotMF = 1\n",
    "fcoor = np.load('MFCr.dat', 'w+')\n",
    "dt = 0.025"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2e-06"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Generate MF Coordinates ##\n",
    "MF_coordinates = np.empty(shape=(numMF, 2))\n",
    "for i in range (1, numMF):\n",
    "    MF_coordinates[i, 1] = np.random.randint(0-Xinstantiate,Longaxis+Xinstantiate)\n",
    "    MF_coordinates[i, 2] = np.random.randint(0-Yinstantiate,Shortaxis+Yinstantiate)\n",
    "    print(fcoor, %d %d\\n % MF_coordinates[i, 1], MF_coordinates[i, 2])\n",
    "fcoor.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "centrex = 750\n",
    "centrey = 150\n",
    "radius = 100\n",
    "finalMF = []\n",
    "finalMF = (MF_coordinates[:, 1]-centrex)^2 + (MF_coordinates[:, 2]-centrey)^2\n",
    "finalMF(finalMF <= radius**2) =1\n",
    "finalMF(finalMF >radius**2) = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Second spatial kernal 1250, 350\n",
    "finalMF1 = []\n",
    "centrex1 = 750\n",
    "centrey1 = 350\n",
    "finalMF = (MF_coordinates[:, 1]-centrex1)^2 + (MF_coordinates[:, 2]-centrey1)^2\n",
    "finalMF(finalMF1 <= radius**2) =1\n",
    "finalMF(finalMF1 >radius**2) = 0\n",
    "find_ac = np.where(finalMF1)\n",
    "finalMF[find_ac] = 1\n",
    "\n",
    "finalMF2=[];\n",
    "centrex2 = 750; %um %center of nw\n",
    "centrey2 = 550;%Shortaxis/2; %um %centre of nw\n",
    "finalMF2 = (MF_coordinates(:,1)-centrex2).^2 +(MF_coordinates(:,2)-centrey2).^2;\n",
    "finalMF2(finalMF2<=radius*radius) = 1;%activated\n",
    "finalMF2(finalMF2>radius*radius)  = 0;%inactivated\n",
    "find_ac = find(finalMF2);\n",
    "finalMF(find_ac)=1;"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
