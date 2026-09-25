"""
建立 Inverted Index（倒排索引）并缓存到硬盘，供 BM25_Ranking.py 和
Test_Fixed.py 共用——跟 corpus_embeddings_cache.pkl 是同一个"算一次、
存起来、以后直接读"的模式，只是这次存的是"词 -> 包含这个词的视频 id 集合"，
不是向量。

【设计决定：用视频 id 当 key，不用行号/位置当 key】
如果用 df 里的行号（position）当索引的 key，一旦以后又用 insert dataframe
的方式往语料库里加新视频（就像你之前加 Rem 视频那样），行号会全部往后挪，
之前存好的索引就全部对不上号了，而且不会报错，会安静地指错视频——这是最
危险的那种 bug。用视频 id 当 key，不管语料库怎么增删、顺序怎么变，只要
id 不变，索引永远查得到正确的视频，更稳妥。
"""
import os
import re
import pickle
import pandas as pd

df = pd.read_parquet("D:/Users/User/Desktop/TikTok_Portfolio/Datasets/test.parquet")

# id 精度修复（跟其余脚本保持一致）
_url_extracted_id = pd.to_numeric(df['url'].str.extract(r'/video/(\d+)')[0], errors='coerce')
_original_id = pd.to_numeric(df['id'], errors='coerce')
df['id'] = _url_extracted_id.fillna(_original_id).astype('Int64')

df["combined_text"] = df["desc"].fillna("") + " " + df["challenges"].fillna("")

def tokenize(text):
    # 跟 BM25_Ranking.py 用同一套分词逻辑，两边才能对得上号
    return re.findall(r"[a-z0-9]+", str(text).lower())

index_path = "D:/Users/User/Desktop/TikTok_Portfolio/Cache/inverted_index_cache.pkl"

print("正在建立 Inverted Index...")
inverted_index = {}
skipped = 0

for video_id, text in zip(df['id'], df['combined_text']):
    if pd.isna(video_id):
        skipped += 1
        continue
    video_id = int(video_id)
    # set() 去重：同一个词在同一篇文档里出现好几次，只需要记一次"这篇文档含这个词"
    for word in set(tokenize(text)):
        inverted_index.setdefault(word, set()).add(video_id)

if skipped > 0:
    print(f"⚠️ 有 {skipped} 行没有可用 id，已跳过，没有被收进索引。")

with open(index_path, "wb") as f:
    pickle.dump(inverted_index, f)

print(f"Inverted Index 建立完成：{len(inverted_index)} 个不同的词，已存到 {index_path}")
print(f"举例——'cosplay' 这个词出现在 {len(inverted_index.get('cosplay', set()))} 篇文档里")
print(f"举例——'rem' 这个词出现在 {len(inverted_index.get('rem', set()))} 篇文档里")
