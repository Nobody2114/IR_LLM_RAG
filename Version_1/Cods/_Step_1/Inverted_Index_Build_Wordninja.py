"""
专门给 Hybrid_LTR.py 用的 Inverted Index——跟给 BM25_Ranking.py /
Hybrid_Ruled.py 用的那份简单索引不同：这份在处理 hashtag 的时候，会先用
wordninja 切词，再把切出来的每个真实单词都记进索引，这样才能正确捕捉到
"remcosplay" 这类拼接 hashtag 里藏着的 "rem" 和 "cosplay"，跟 Hybrid_LTR.py
自己 extract_soft_features() 里的判断逻辑保持一致。

【重要】不要把这份索引拿去给 BM25_Ranking.py / Hybrid_Ruled.py 用——那两份
刻意保留简单分词、不做 wordninja 切词，是为了让"有没有精细分词"成为一个
干净的对照变量，方便量化 wordninja 到底带来多少提升。三个方法都用上这份
索引，会让这个对照变量消失，比较的意义会从"细致工程 vs 简陋baseline"
变成完全不同的问题。
"""
import os
import re
import ast
import pickle
import pandas as pd
import wordninja

df = pd.read_parquet("D:/Users/User/Desktop/TikTok_Portfolio/Datasets/test.parquet")

_url_extracted_id = pd.to_numeric(df['url'].str.extract(r'/video/(\d+)')[0], errors='coerce')
_original_id = pd.to_numeric(df['id'], errors='coerce')
df['id'] = _url_extracted_id.fillna(_original_id).astype('Int64')

df["combined_text"] = df["desc"].fillna("") + " " + df["challenges"].fillna("")
df["challenges_str"] = df["challenges"].fillna("").astype(str)

def parse_hashtag_tokens(challenges_str):
    try:
        tokens = ast.literal_eval(str(challenges_str))
        if isinstance(tokens, (list, tuple)):
            return [str(t).lower() for t in tokens]
    except (ValueError, SyntaxError):
        pass
    return [str(challenges_str).lower()]

_segment_cache = {}
def segment_hashtag(tok):
    if tok not in _segment_cache:
        _segment_cache[tok] = [w.lower() for w in wordninja.split(tok)]
    return _segment_cache[tok]

def words_for_row(combined_text, challenges_str):
    words = set()
    # 正文/自由文本里独立出现的词（简单正则分词，逼近 has_word() 的整词边界匹配）
    words.update(re.findall(r"[a-z0-9]+", str(combined_text).lower()))
    # hashtag 先切词，再把切出来的每个真实单词记进去（对应 wordninja 那条判断路径）
    for tok in parse_hashtag_tokens(challenges_str):
        words.update(segment_hashtag(tok))
    return words

index_path = "D:/Users/User/Desktop/TikTok_Portfolio/Cache/inverted_index_wordninja_cache.pkl"

print("正在建立 wordninja 版 Inverted Index...")
inverted_index = {}
skipped = 0

for video_id, text, challenges in zip(df['id'], df['combined_text'], df['challenges_str']):
    if pd.isna(video_id):
        skipped += 1
        continue
    video_id = int(video_id)
    for word in words_for_row(text, challenges):
        inverted_index.setdefault(word, set()).add(video_id)

if skipped > 0:
    print(f"⚠️ 有 {skipped} 行没有可用 id，已跳过。")

with open(index_path, "wb") as f:
    pickle.dump(inverted_index, f)

print(f"wordninja 版 Inverted Index 建立完成：{len(inverted_index)} 个不同的词，已存到 {index_path}")
print(f"举例——'rem' 出现在 {len(inverted_index.get('rem', set()))} 篇文档里")
print(f"举例——'cosplay' 出现在 {len(inverted_index.get('cosplay', set()))} 篇文档里")
