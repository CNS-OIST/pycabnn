

```python
import sqlite3
```


```python
conn = sqlite3.connect('test.db')
```


```python
c = conn.cursor()
```


```python
c.execute('DROP TABLE IF EXISTS ' + 'stocks')
c.execute('''CREATE TABLE stocks
             (date text, trans text, symbol text, qty int, price real)''')

# Insert a row of data
c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.

```


```python
t = ('BUY',)
c.execute('SELECT date FROM stocks WHERE trans=?', t)
c.fetchall()
```




    [('2006-01-05',)]




```python
conn.close()
```


```python
purchases = [('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
             ('2006-04-05', 'BUY', 'MSFT', 1000, 72.00),
             ('2006-04-06', 'SELL', 'IBM', 500, 53.00),
            ]
c.executemany('INSERT INTO stocks VALUES (?,?,?,?,?)', purchases)
```




    <sqlite3.Cursor at 0x10e112110>




```python
c.execute('SELECT MIN(qty) FROM stocks')
c.fetchone()
```




    (100,)




```python
z = ('TEST' +
'HELLO')
z
```




    'TESTHELLO'




```python
name = 'connection'
prefix = 'pf_goc.db'
debug = True

import sqlite3
conn = sqlite3.connect(prefix)
c = conn.cursor()
c.execute('DROP TABLE IF EXISTS ' + name)
command = ('CREATE TABLE ' + name +
      '(source int, target int, segment int, branch int, distance real')

if debug:
    command = command + ', x real, y real, z real)'
else:
    command = command + ')'

c.execute(command)
conn.commit()
conn.close()

```


```python
!open .
```


```python
import numpy as np
import pandas as pd
import sqlite3
```


```python
for k in range(120):
    x = np.loadtxt('/Users/shhong/Dropbox/network_data/output_brep_2/PFtoGoCtargets%d.dat' % k).astype(int)
    y = np.arange(k, 1995, 120)
    print(x.max(), y.max())
```

    1920 1920
    1921 1921
    1922 1922
    1923 1923
    1924 1924
    1925 1925
    1926 1926
    1927 1927
    1928 1928
    1929 1929
    1930 1930
    1931 1931
    1932 1932
    1933 1933
    1934 1934
    1935 1935
    1936 1936
    1937 1937
    1938 1938
    1939 1939
    1940 1940
    1941 1941
    1942 1942
    1943 1943
    1944 1944
    1945 1945
    1946 1946
    1947 1947
    1948 1948
    1949 1949
    1950 1950
    1951 1951
    1952 1952
    1953 1953
    1954 1954
    1955 1955
    1956 1956
    1957 1957
    1958 1958
    1959 1959
    1960 1960
    1961 1961
    1962 1962
    1963 1963
    1964 1964
    1965 1965
    1966 1966
    1967 1967
    1968 1968
    1969 1969
    1970 1970
    1971 1971
    1972 1972
    1973 1973
    1974 1974
    1975 1975
    1976 1976
    1977 1977
    1978 1978
    1979 1979
    1980 1980
    1981 1981
    1982 1982
    1983 1983
    1984 1984
    1985 1985
    1986 1986
    1987 1987
    1868 1988
    1989 1989
    1990 1990
    1991 1991
    1872 1992
    1993 1993
    1994 1994
    1875 1875
    1876 1876
    1877 1877
    1878 1878
    1879 1879
    1880 1880
    1881 1881
    1882 1882
    1883 1883
    1884 1884
    1885 1885
    1886 1886
    1887 1887
    1888 1888
    1889 1889
    1890 1890
    1891 1891
    1892 1892
    1893 1893
    1894 1894
    1895 1895
    1896 1896
    1897 1897
    1898 1898
    1899 1899
    1900 1900
    1901 1901
    1902 1902
    1903 1903
    1904 1904
    1905 1905
    1906 1906
    1907 1907
    1908 1908
    1909 1909
    1910 1910
    1911 1911
    1912 1912
    1913 1913
    1914 1914
    1915 1915
    1916 1916
    1917 1917
    1918 1918
    1919 1919



```python
i = 0

```




    array([   0,  120,  240,  360,  480,  600,  720,  840,  960, 1080, 1200,
           1320, 1440, 1560, 1680, 1800, 1920])




```python
db = '/Users/shhong/Dropbox/network_data/output_pybrep_2/AAtoGoC.db'
Path(db).parent
ntargets = 1995
nblocks = 120

# def run_block(i):
i = 0

```


```python
z = np.vstack([df['segment'].values, df['branch'].values])
np.savetxt('test.dat', z.T, delimiter=' ', fmt='%d')

```


```python
from pathlib import Path
Path(db).parent
```




    PosixPath('/Users/shhong/Dropbox/network_data/output_pybrep_2')




```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>source</th>
      <th>target</th>
      <th>segment</th>
      <th>branch</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>490368</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>663466</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>454240</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>476200</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>491550</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>478713</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>414755</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>293696</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>164247</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>65068</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>227504</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>127210</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>372916</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>574846</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>464833</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>357221</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>85411</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>190049</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>398847</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>582317</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>599297</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>408855</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>384360</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>180162</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>100365</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>154878</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>180768</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>107321</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>22511</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>174556</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115980</th>
      <td>61069</td>
      <td>764037</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>115981</th>
      <td>61070</td>
      <td>210770</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>115982</th>
      <td>61071</td>
      <td>387498</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>115983</th>
      <td>61072</td>
      <td>430176</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>115984</th>
      <td>61073</td>
      <td>630748</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>115985</th>
      <td>61074</td>
      <td>315209</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>115986</th>
      <td>61075</td>
      <td>639400</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>115987</th>
      <td>61076</td>
      <td>613779</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>115988</th>
      <td>61077</td>
      <td>344991</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>115989</th>
      <td>61078</td>
      <td>86751</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>115990</th>
      <td>61079</td>
      <td>131721</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>115991</th>
      <td>61080</td>
      <td>503815</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>115992</th>
      <td>61081</td>
      <td>274994</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>115993</th>
      <td>61082</td>
      <td>87091</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>115994</th>
      <td>61083</td>
      <td>369522</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>115995</th>
      <td>61084</td>
      <td>22672</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>115996</th>
      <td>61085</td>
      <td>548491</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>115997</th>
      <td>61086</td>
      <td>217810</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>115998</th>
      <td>61087</td>
      <td>346289</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>115999</th>
      <td>61088</td>
      <td>80344</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>116000</th>
      <td>61089</td>
      <td>419280</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>116001</th>
      <td>61090</td>
      <td>775959</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>116002</th>
      <td>61091</td>
      <td>678248</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>116003</th>
      <td>61092</td>
      <td>312156</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>116004</th>
      <td>61093</td>
      <td>503075</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>116005</th>
      <td>61094</td>
      <td>432665</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>116006</th>
      <td>61095</td>
      <td>740654</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>116007</th>
      <td>61096</td>
      <td>411883</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>116008</th>
      <td>61097</td>
      <td>81188</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>116009</th>
      <td>61098</td>
      <td>703336</td>
      <td>1920</td>
      <td>3</td>
      <td>2</td>
      <td>73.0</td>
    </tr>
  </tbody>
</table>
<p>116010 rows × 6 columns</p>
</div>




```python
Path(db)
```




    PosixPath('/Users/shhong/Dropbox/network_data/output_pybrep_2/AAtoGoC.db')




```python
db
```




    '/Users/shhong/Dropbox/network_data/output_pybrep_2/AAtoGoC.db'


