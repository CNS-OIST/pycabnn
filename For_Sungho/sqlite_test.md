

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
