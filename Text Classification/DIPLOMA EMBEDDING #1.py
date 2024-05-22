#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('medium_articles.csv')


# In[3]:


df.head()


# In[4]:


df_tagged = df


# In[5]:


df.drop(columns=['tags', 'timestamp', 'authors'])


# In[6]:


df['text'][0]


# In[7]:


df_short= df.sample(n=100000, random_state=42)


# In[8]:


df_short.head()


# In[9]:


df_short.describe()


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re 
import nltk
import spacy 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# In[11]:


## clean the data 
stopwordSet = set(stopwords.words("english"))

## NlP Processing
lemma = WordNetLemmatizer()

def cleanup_sentences(sentence):
    text = re.sub('[^a-zA-Z]'," ", sentence) # Removing non a-z characters
    text = text.lower() # Lowering all text
    text = word_tokenize(text, language="english") # Splitting each word into an element of a list
    text = [lemma.lemmatize(word) for word in text if(word) not in stopwordSet] # Lemmatizing words and removing stopwords
    text = " ".join(text) # Putting words back into a single string. ['the', 'brown', 'cow'] --> 'the brown cow'
    return text

## apply the function to the data 
df_short['text_cleaned'] = df_short['text'].apply(cleanup_sentences)


# In[12]:


df_short.reset_index(drop=True, inplace=True)


# In[13]:


df_short['text_cleaned']


# In[14]:


get_ipython().run_cell_magic('capture', '', '!pip install git+https://github.com/MaartenGr/BERTopic.git@master\n\n!pip install cudf-cu12 dask-cudf-cu12 --extra-index-url=https://pypi.nvidia.com\n!pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com\n!pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com\n!pip install cupy-cuda12x -f https://pip.cupy.dev/aarch64\n\n!pip install safetensors\n!pip install datasets\n!pip install datashader\n!pip install adjustText\n')


# In[15]:


df_short.reset_index(drop=True, inplace=True)
texts_only = df_short['text_cleaned']; len(texts_only)


# In[16]:


texts_only.head()


# ## 1. Sentence transformer embedding

# In[17]:


# from sentence_transformers import SentenceTransformer
# import numpy as np
# # Create embeddings
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(texts_only, show_progress_bar=True)

# np.save('embeddings.npy', embeddings)


# ## 2. Distilled Models (DistilBERT with distillation):

# In[18]:


from sentence_transformers import SentenceTransformer

# Load DistilBERT model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Use the model for encoding
embeddings = model.encode(texts_only)


# ## 3. Word2Vec embeddings

# In[20]:


# from gensim.models import Word2Vec

# # Train word2vec model on your own data
# sentences = [text.split() for text in texts_only]
# model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# # Use the trained model for encoding
# embeddings = [model.wv[word] for word in sentences[0]]


# In[19]:


# import tensorflow as tf
# from transformers import BertModel, BertTokenizer

# # Load pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Quantize the model
# quantized_model = tf.lite.TFLiteConverter.from_keras_model(model)
# quantized_model.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = quantized_model.convert()

# # Save the quantized model to file
# with open('quantized_bert_model.tflite', 'wb') as f:
#     f.write(tflite_model)


# In[22]:


import collections
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

# Extract vocab to be used in BERTopic
vocab = collections.Counter()
tokenizer = CountVectorizer().build_tokenizer()
for i in tqdm(texts_only):
  vocab.update(tokenizer(i))
vocab = [word for word, frequency in vocab.items() if frequency >= 15]; len(vocab)


# In[49]:


import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from scipy.linalg import eigh

# Convert embeddings to a dense array and normalize them
dense_embeddings = np.array(embeddings)
normalized_embeddings = dense_embeddings / np.linalg.norm(dense_embeddings, axis=1, keepdims=True)

# Adjust UMAP parameters
umap_model = UMAP(n_components=2, n_neighbors=100, random_state=42, metric="cosine")
reduced_embeddings = umap_model.fit_transform(normalized_embeddings)

# Adjust HDBSCAN parameters for more balanced clusters
hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True, min_cluster_size=10)
clusters = hdbscan_model.fit(reduced_embeddings).labels_

print(clusters)


# In[99]:


plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=5)  # Scatter plot
plt.title('UMAP Visualization')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()


# In[101]:


umap_model_1 = UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric='euclidean')
umap_embeddings = umap_model_1.fit_transform(reduced_embeddings)

# Step 2: Plot the reduced embeddings
plt.figure(figsize=(10, 8))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=5)  # Scatter plot
plt.title('UMAP Visualization')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()


# In[50]:


from sentence_transformers import SentenceTransformer
from bertopic.cluster import BaseCluster
from bertopic.representation import KeyBERTInspired


# In[51]:


# Fit BERTopic without actually performing any clustering
topic_model= BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model).fit(texts_only, embeddings=reduced_embeddings, y=clusters)


# In[52]:


topic_model.get_topic_info()


# In[53]:


import itertools
import pandas as pd

# Define colors for the visualization to iterate over
colors = itertools.cycle(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'])
color_key = {str(topic): next(colors) for topic in set(topic_model.topics_) if topic != -1}

# Prepare dataframe and ignore outliers
df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1], "Topic": [str(t) for t in topic_model.topics_]})
df["Length"] = [len(texts_only) for doc in texts_only]
df = df.loc[df.Topic != "-1"]
df = df.loc[(df.y > -10) & (df.y < 10) & (df.x < 10) & (df.x > -10), :]
df["Topic"] = df["Topic"].astype("category")

# Get centroids of clusters
mean_df = df.groupby("Topic").mean().reset_index()
mean_df.Topic = mean_df.Topic.astype(int)
mean_df = mean_df.sort_values("Topic")


# In[54]:


import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe

fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111)  # Create subplot
sns.scatterplot(data=df, x='x', y='y', c=df['Topic'].map(color_key), alpha=0.4, sizes=(0.4, 10), size="Length", ax=ax)

# Annotate top 50 topics
texts, xs, ys = [], [], []
for row in mean_df.iterrows():
    topic = row[1]["Topic"]
    name = " - ".join(list(zip(*topic_model.get_topic(int(topic))))[0][:3])

    if int(topic) <= 30:
        xs.append(row[1]["x"])
        ys.append(row[1]["y"])
        texts.append(ax.text(row[1]["x"], row[1]["y"], name, size=10, ha="center", color=color_key[str(int(topic))],
                             path_effects=[pe.withStroke(linewidth=0.5, foreground="black")]))

# Adjust annotations such that they do not overlap
adjust_text(texts, x=xs, y=ys, time_lim=1, force_text=(0.01, 0.02), force_static=(0.01, 0.02), force_pull=(0.5, 0.5))

# Center the plot
ax.set_xlim([min(xs) - 0.1, max(xs) + 0.1])  # Add some padding to the x-axis
ax.set_ylim([min(ys) - 0.1, max(ys) + 0.1])  # Add some padding to the y-axis

plt.show()


# In[55]:


topic_model.visualize_barchart()


# In[56]:


topic_model.get_topic_freq().head(20)


# In[57]:


topic_model.visualize_hierarchy(top_n_topics=20)


# In[67]:


topic_model.get_topic(similar_topics[0])


# In[68]:


topic_model.get_topic(similar_topics[1])


# In[59]:


topic_model.get_topic(similar_topics[2])


# In[60]:


for i in range(10):
    print(topic_model.get_topic(i))


# In[61]:


topic_model.visualize_heatmap()


# In[69]:


clusters


# In[70]:


max(clusters)


# ## Grouping smaller clusters into larger ones

# In[77]:


# Ensure all elements in clusters are non-negative
clusters = np.array(clusters)
clusters[clusters < 0] = 10000  # Set negative values to a large positive value
min_cluster_size = 100

# Now you can analyze clusters and identify small clusters
cluster_counts = np.bincount(clusters)
small_clusters_indices = np.where(cluster_counts <= min_cluster_size)[0]

# Merge small clusters into larger ones
for idx in small_clusters_indices:
    if idx >= 0:  # Skip sentinel values
        if np.any(clusters == idx):  # Check if there are elements satisfying the condition
            nearest_cluster_idx = np.argmin(np.linalg.norm(reduced_embeddings - reduced_embeddings[clusters == idx].mean(axis=0), axis=1))
            if np.any(clusters == nearest_cluster_idx):  # Check if there are elements satisfying the condition
                for i in range(len(clusters)):
                    if clusters[i] == idx:
                        clusters[i] = nearest_cluster_idx

# Refit BERTopic model with updated cluster labels
topic_model.fit(texts_only, embeddings=reduced_embeddings, y=clusters)


# In[78]:


topic_model.visualize_hierarchy(top_n_topics=20)


# In[79]:


topic_model.visualize_heatmap()


# In[81]:


topic_model.get_topic_freq().count()


# In[88]:


clusters


# In[86]:


topic_model.get_topic_freq().tail(20)


# In[89]:


# Ensure all elements in clusters are non-negative
clusters = np.array(clusters)
clusters[clusters < 0] = 10000  # Set negative values to a large positive value
min_cluster_size = 100

# Now you can analyze clusters and identify small clusters
cluster_counts = np.bincount(clusters)
small_clusters_indices = np.where(cluster_counts <= min_cluster_size)[0]

# Merge small clusters into larger ones
for idx in small_clusters_indices:
    if idx >= 0:  # Skip sentinel values
        nearest_cluster_idx = np.argmax(cluster_counts)  # Find the cluster with the largest count
        clusters[clusters == idx] = nearest_cluster_idx

# Refit BERTopic model with updated cluster labels
topic_model.fit(texts_only, embeddings=reduced_embeddings, y=clusters)


# In[90]:


topic_model.get_topic_freq().head(20)


# In[91]:


topic_model


# In[92]:


topic_model.get_topic_freq().count()


# In[93]:


# import numpy as np
# from sklearn.cluster import KMeans
# from umap import UMAP
# from hdbscan import HDBSCAN
# from bertopic import BERTopic

# # Ensure all elements in clusters are non-negative
# clusters = np.array(clusters)
# clusters[clusters < 0] = 10000  # Set negative values to a large positive value
# min_cluster_size = 100

# # Analyze clusters and identify small clusters
# cluster_counts = np.bincount(clusters)
# small_clusters_indices = np.where(cluster_counts <= min_cluster_size)[0]

# # Get embeddings of small clusters
# small_clusters_mask = np.isin(clusters, small_clusters_indices)
# small_clusters_embeddings = reduced_embeddings[small_clusters_mask]

# # Perform K-means on small clusters
# num_kmeans_clusters = 100  # Set the desired number of clusters for small clusters
# kmeans = KMeans(n_clusters=num_kmeans_clusters, random_state=42)
# kmeans_labels = kmeans.fit_predict(small_clusters_embeddings)

# # Replace small cluster labels with new K-means cluster labels
# new_cluster_label = max(clusters) + 1
# for small_cluster_index, kmeans_label in zip(small_clusters_indices, kmeans_labels):
#     clusters[clusters == small_cluster_index] = new_cluster_label + kmeans_label

# # Refit BERTopic model with updated cluster labels
# topic_model = BERTopic(
#     embedding_model=embedding_model,
#     umap_model=umap_model,
#     hdbscan_model=hdbscan_model,
#     vectorizer_model=vectorizer_model,
#     representation_model=representation_model
# ).fit(texts_only, embeddings=reduced_embeddings, y=clusters)


# In[94]:


# Ensure all elements in clusters are non-negative
clusters = np.array(clusters)
clusters[clusters < 0] = 10000  # Set negative values to a large positive value
min_cluster_size = 100

# Analyze clusters and identify small clusters
cluster_counts = np.bincount(clusters)
small_clusters_indices = np.where(cluster_counts <= min_cluster_size)[0]


# In[96]:


# Check if there are small clusters to process
if len(small_clusters_indices) > 0:
    # Get embeddings of small clusters
    small_clusters_mask = np.isin(clusters, small_clusters_indices)
    small_clusters_embeddings = reduced_embeddings[small_clusters_mask]

    # Ensure there are samples in small_clusters_embeddings
    if small_clusters_embeddings.shape[0] > 0:
        # Perform K-means on small clusters
        num_kmeans_clusters = 100  # Set the desired number of clusters for small clusters
        kmeans = KMeans(n_clusters=num_kmeans_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(small_clusters_embeddings)

        # Replace small cluster labels with new K-means cluster labels
        new_cluster_label = max(clusters) + 1
        for small_cluster_index, kmeans_label in zip(small_clusters_indices, kmeans_labels):
            clusters[clusters == small_cluster_index] = new_cluster_label + kmeans_label

# Refit BERTopic model with updated cluster labels
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model
).fit(texts_only, embeddings=reduced_embeddings, y=clusters)


# In[97]:


topic_model.get_topic_info()


# In[98]:


from sklearn.metrics import silhouette_score

sil_score = silhouette_score(reduced_embeddings, clusters)
print("Silhouette Score: ", sil_score)


# In[41]:


topic_model.get_topic_info()


# In[44]:


from sklearn.metrics import silhouette_score

sil_score = silhouette_score(reduced_embeddings, clusters)
print("Silhouette Score: ", sil_score)


# In[107]:


topic_model.reduce_topics(texts_only, nr_topics=30)


# In[108]:


topic_model.get_topic_info()


# In[109]:


topic_model.reduce_outliers(texts_only, topics)


# In[110]:


topic_model.get_topic_info()


# In[115]:


from bertopic import BERTopic
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd

docs = texts_only

topic_model = BERTopic(verbose=True, n_gram_range=(1, 3))
topics, _ = topic_model.fit_transform(docs)

# Preprocess Documents
documents = pd.DataFrame({"Document": docs,
                          "ID": range(len(docs)),
                          "Topic": topics})
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

# Extract vectorizer and analyzer from BERTopic
vectorizer = topic_model.vectorizer_model
analyzer = vectorizer.build_analyzer()

# Extract features for Topic Coherence evaluation
words = vectorizer.get_feature_names_out()
tokens = [analyzer(doc) for doc in cleaned_docs]
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]
topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
               for topic in range(len(set(topics))-1)]

# Evaluate
coherence_model = CoherenceModel(topics=topic_words, 
                                 texts=tokens, 
                                 corpus=corpus,
                                 dictionary=dictionary, 
                                 coherence='c_v')
coherence = coherence_model.get_coherence()
print(coherence)


# In[116]:


topic_model.get_topic_info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## -----------------------------------------------------------------------------------

# In[ ]:


from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Train UMAP model
umap_model = UMAP(n_components=5, n_neighbors=30, min_dist=0.0, random_state=42)

# Train HDBSCAN model
hdbscan_model = HDBSCAN(min_cluster_size=100, min_samples=10, cluster_selection_epsilon=0.1)

# Use a custom tokenizer with CountVectorizer for better text preprocessing
custom_tokenizer = CountVectorizer().build_analyzer()
vectorizer_model = CountVectorizer(tokenizer=custom_tokenizer, min_df=5, stop_words="english")

# Initialize BERTopic model
topic_model= BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model).fit(texts_only, embeddings=embeddings, y=clusters)

# Fit BERTopic model
topics, probabilities = topic_model.fit_transform(texts_only)

# Get topic information
topic_info = topic_model.get_topic_info()

# Display topic information
print(topic_info)


# In[ ]:


topic_model.get_topic_info()


# In[ ]:


topic_model.visualize_hierarchy(top_n_topics=40)


# In[ ]:


topic_info["Representation"][0]


# In[ ]:


embedding_model_structure = topic_model.embedding_model
umap_model_structure = topic_model.umap_model
hdbscan_model_structure = topic_model.hdbscan_model
vectorizer_model_structure = topic_model.vectorizer_model
representation_model_structure = topic_model.representation_model

print(embedding_model_structure)


# In[ ]:


print(umap_model_structure)


# In[ ]:


print(hdbscan_model_structure)


# In[ ]:


print(vectorizer_model_structure)


# In[ ]:


print(representation_model_structure)


# In[ ]:


df_short.reset_index(drop=True, inplace=True)
texts_only = df_short['text_cleaned']; len(texts_only)


# In[ ]:


texts = df_short['text_cleaned']
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import DBSCAN
import pandas as pd


# Initialize embedding model
embedding_model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

# Encode texts using embedding model
embeddings = embedding_model.encode(texts)

# Reduce dimensionality with UMAP
umap_model = UMAP(n_components=50, n_neighbors=30, min_dist=0.0, random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)

# Cluster with DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(umap_embeddings)

# Print results
print("Cluster Labels:", labels)


# In[ ]:


texts


# In[ ]:


texts_only


# In[ ]:




