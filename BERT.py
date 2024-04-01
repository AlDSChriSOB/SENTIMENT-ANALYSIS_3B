#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd


# In[2]:


# ! pip install "tensorflow>=2.0.0"
# ! pip install --upgrade tensorflow-hub


# In[3]:


# ! pip install tensorflow-text


# In[11]:


df = pd.read_csv('IMDB Dataset.csv')
df.head()


# In[12]:


df['sentiment'].value_counts()


# ### Adding labels

# In[13]:


df['positive'] = df['sentiment'].apply(lambda x:1 if x=='positive' else 0)


# In[14]:


df.head()


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['positive'], stratify=df['positive'])


# ### Getting started with BERT

# In[16]:


# Downloading the BERT model

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[17]:


# Initializing the BERT layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)


# In[20]:


# Initializing the neural network layers

l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)


# In[22]:


model = tf.keras.Model(inputs=[text_input], outputs = [l])

model.summary()


# ### Model compling

# In[23]:


METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
 loss='binary_crossentropy',
 metrics=METRICS)


# ### Fitting the model

# In[24]:


model.fit(X_train, y_train, epochs=5)


# ### Evaluating the model

# In[25]:


y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()


# In[ ]:





# ### References
# https://www.section.io/engineering-education/classification-model-using-bert-and-tensorflow/
