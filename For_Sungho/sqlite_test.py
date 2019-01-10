
# coding: utf-8

# In[36]:


import sqlite3


# In[37]:


conn = sqlite3.connect('test.db')


# In[38]:


c = conn.cursor()


# In[41]:


c.execute('DROP TABLE IF EXISTS ' + 'stocks')
c.execute('''CREATE TABLE stocks
             (date text, trans text, symbol text, qty int, price real)''')

# Insert a row of data
c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.


# In[30]:


t = ('BUY',)
c.execute('SELECT date FROM stocks WHERE trans=?', t)
c.fetchall()


# In[35]:


conn.close()


# In[31]:


purchases = [('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
             ('2006-04-05', 'BUY', 'MSFT', 1000, 72.00),
             ('2006-04-06', 'SELL', 'IBM', 500, 53.00),
            ]
c.executemany('INSERT INTO stocks VALUES (?,?,?,?,?)', purchases)


# In[34]:


c.execute('SELECT MIN(qty) FROM stocks')
c.fetchone()


# In[43]:


z = ('TEST' +
'HELLO')
z


# In[47]:


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


# In[46]:


get_ipython().system('open .')

