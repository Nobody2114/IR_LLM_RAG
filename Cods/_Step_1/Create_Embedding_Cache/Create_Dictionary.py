#这个code是用来 创造ML 的 dictionary，那么下一次就不需要重新进行NLP + Vectorization ( 这个 process 也可以叫做 Offline Indexing)
import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login

df = pd.read_parquet("D:/Users/User/Desktop/TikTok_Portfolio/Datasets/test.parquet")
login(token="#")


"""
这个是第一版本, 只是reading file 里面的 challenges, 但是challenges 就是 hashtag, 不会拥有丰富的上下文, 那么就需要下面那个版本
corpus = df["challenges"].astype(str).tolist()
"""

# 1. 准备文本数据
df["combined_text"] = df["desc"].fillna("") + " " + df["challenges"].fillna("")
corpus = df["combined_text"].astype(str).tolist()


# 3. 加载模型（这行必须保留，因为需要模型来把用户的 query 也转成向量）
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 4. 【核心优化】检查本地是否已经有缓存的向量文件
embedding_file = "corpus_embeddings_cache.pkl"

if os.path.exists(embedding_file):
    print("Loading pre-computed embeddings from local file...")
    with open(embedding_file, "rb") as f:
        corpus_embeddings = pickle.load(f)
else:
    print("Encoding corpus for the first time...")
    # 第一次运行会执行这里（比较慢）
    corpus_embeddings = model.encode(corpus, show_progress_bar=True)
    
    # 保存到本地文件，供下一次直接使用
    with open(embedding_file, "wb") as f:
        pickle.dump(corpus_embeddings, f)
    print("Embeddings saved to local cache!")
