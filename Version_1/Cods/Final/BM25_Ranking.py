"""
BM25 —— 经典的纯词汇检索排序方法（Elasticsearch/Lucene 底层默认用的公式）。
跟 Twist.py（Dense Vector + LTR）、Test_Fixed.py（Dense Vector + Boolean）
最根本的差异：BM25 完全不涉及语义向量/embedding，纯粹靠词频统计打分——
一个词在某篇文档里出现得越多、但在整个语料库里又足够罕见，这篇文档的
分数就越高，同时会用文档长度做归一化（避免长文档单纯靠字数堆砌占便宜）。

这里刻意保持"纯 BM25"，不混入 similarity_score、engagement_score 或任何
手写加权——这样才是一个干净、有教学意义的经典 IR baseline，方便跟
Dense Vector 系的方法做架构层面的对比，而不是又做出一个"缝合怪"。
"""
import os
import re
import pickle
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score

df = pd.read_parquet("D:/Users/User/Desktop/TikTok_Portfolio/Datasets/test.parquet")

# 【id 精度修复】跟其余脚本保持一致：优先信任从 url 反推出的精确 id。
_url_extracted_id = pd.to_numeric(df['url'].str.extract(r'/video/(\d+)')[0], errors='coerce')
_original_id = pd.to_numeric(df['id'], errors='coerce')
df['id'] = _url_extracted_id.fillna(_original_id).astype('Int64')

df["combined_text"] = df["desc"].fillna("") + " " + df["challenges"].fillna("")
corpus = df["combined_text"].astype(str).tolist()

queries = [
    "Rem cosplay",
    "Genshin Cosplay",
    "Logo Design",
]

# === 分词 ===
# BM25 需要先把每篇文档、每个 query 切成一串词（token）。这里刻意用最简单
# 的正则分词（小写化、只保留字母数字），不做 wordninja 拼接词切分——
# 这正是 BM25 baseline 该有的样子：如果语料库里的 hashtag 是拼在一起的
# "remcosplay"，BM25 会把它当成一个独立的词，跟单独的 "rem"、"cosplay"
# 是三个不同的词，天生识别不出拼接关系。这不是 bug，是纯词汇方法本身
# 的局限，值得在报告里如实呈现，而不是悄悄修掉。
def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())

tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
# 注意：建立 BM25Okapi 这一步本身，仍然需要看过全部语料（要算每个词的
# IDF、平均文档长度这些全局统计量），这一步省不掉。Inverted Index 真正
# 帮上忙的地方，是接下来"对某个 query，只计算一小撮候选文档的分数"，
# 而不是每次查询都对全部文档跑一遍 get_scores()。

# 【新增】读取 Inverted Index 缓存
index_path = "D:/Users/User/Desktop/TikTok_Portfolio/Cache/inverted_index_cache.pkl"
with open(index_path, "rb") as f:
    inverted_index = pickle.load(f)

# rank_bm25 内部是用"文档在语料库列表里的位置"（0, 1, 2...）来认文档的，
# 不是用视频 id。Inverted Index 缓存里存的是 id（原因见 Inverted_Index_
# Build.py 的说明），所以这里需要一张"id -> 位置"的对照表，把从索引里
# 查到的 id，翻译成 rank_bm25 认得的位置，才能调用 get_batch_scores()。
# 这张表只需要在内存里现算，很快（跟语料库大小同阶），不需要额外缓存。
id_to_pos = {int(vid): pos for pos, vid in enumerate(df['id']) if pd.notna(vid)}

all_query_dfs = []

for q_id, query_text in enumerate(queries):
    tokenized_query = tokenize(query_text)

    # 【核心改动】不再对全部语料调用 get_scores()，而是先用 Inverted Index
    # 查出"至少包含一个查询词"的候选 id 集合（这一步只是查表，飞快），
    # 再只对这一小撮候选算 BM25 分数——这才是 Inverted Index 真正该发挥
    # 作用的地方，也是真实搜索引擎（Elasticsearch/Lucene）的标准做法。
    candidate_ids = set()
    for word in tokenized_query:
        candidate_ids |= inverted_index.get(word, set())

    if not candidate_ids:
        print(f"⚠️ [{query_text}] Inverted Index 里一个候选都没查到，跳过。")
        continue

    candidate_positions = [id_to_pos[vid] for vid in candidate_ids if vid in id_to_pos]
    batch_scores = bm25.get_batch_scores(tokenized_query, candidate_positions)

    temp_df = df.iloc[candidate_positions].copy()
    temp_df["query_id"] = q_id
    temp_df["bm25_score"] = batch_scores

    # 候选池上限跟其余脚本保持一致（1500），排序信号只用 bm25_score
    candidate_df = temp_df[temp_df["bm25_score"] > 0].copy()  # 分数为0代表一个词都没匹配上，直接排除
    if len(candidate_df) == 0:
        print(f"⚠️ [{query_text}] BM25 一条候选都没有，跳过。")
        continue
    candidate_df = candidate_df.nlargest(min(1500, len(candidate_df)), "bm25_score")
    all_query_dfs.append(candidate_df)

master_df = pd.concat(all_query_dfs, ignore_index=True)

# === Ground truth（跟 Twist.py / Test_Fixed.py / Combine_Tuple.py 一致）===
GROUND_TRUTH_FILES = {
    "Rem cosplay":     "D:/Users/User/Desktop/TikTok_Portfolio/Ground_Truth/Global_GT/Backup/Rem_Ground_Truth_B.xlsx",
    "Genshin Cosplay": "D:/Users/User/Desktop/TikTok_Portfolio/Ground_Truth/Global_GT/Backup/Genshin_Ground_Truth_B.xlsx",
    "Logo Design":     "D:/Users/User/Desktop/TikTok_Portfolio/Ground_Truth/Global_GT/Backup/Logo_Ground_Truth_B.xlsx",
}

def load_ground_truth(path):
    gt = pd.read_excel(path)
    gt['id'] = gt['url'].str.extract(r'/video/(\d+)').astype('int64')
    label_col = 'relevance_label' if 'relevance_label' in gt.columns else 'relevant'
    gt = gt.rename(columns={label_col: 'true_relevance'})
    gt['true_relevance'] = pd.to_numeric(gt['true_relevance'], errors='coerce')
    bad_rows = gt[gt['true_relevance'].isna()]
    if len(bad_rows) > 0:
        print(f"⚠️ {path} 里有 {len(bad_rows)} 行 true_relevance 不是纯数字：")
        print(bad_rows[['id', 'url']].to_string())
        gt = gt.dropna(subset=['true_relevance'])
    return gt[['id', 'true_relevance']]

# === 评估：纯 BM25 打分，不训练任何模型，不需要 leave-one-query-out ===
K = 20
eval_summary = []

for query_text, gt_path in GROUND_TRUTH_FILES.items():
    if query_text not in queries or not os.path.exists(gt_path):
        print(f"⚠️ 跳过 [{query_text}]")
        continue

    gt_query_id = queries.index(query_text)
    ground_truth = load_ground_truth(gt_path)

    if gt_query_id not in master_df['query_id'].unique():
        print(f"⚠️ 跳过 [{query_text}]：候选生成阶段没有产出任何候选。")
        continue

    eval_df = master_df[master_df['query_id'] == gt_query_id].copy()
    eval_df = eval_df.dropna(subset=['id'])
    eval_df['id'] = eval_df['id'].astype('int64')

    # --- 漏召回诊断 ---
    missing_ids = set(ground_truth['id']) - set(eval_df['id'])
    missing_df = ground_truth[ground_truth['id'].isin(missing_ids)]
    relevant_gt = ground_truth[ground_truth['true_relevance'] >= 3]
    relevant_missing = missing_df[missing_df['true_relevance'] >= 3]
    true_recall = (1 - len(relevant_missing) / len(relevant_gt)) if len(relevant_gt) > 0 else float('nan')

    print(f"=== [{query_text}] BM25 召回诊断 ===")
    print(f"漏召回总数: {len(missing_df)}")
    print(f"其中 true_relevance = 0: {(missing_df['true_relevance'] == 0).sum()}")
    print(f"其中 true_relevance >= 3: {(missing_df['true_relevance'] >= 3).sum()}")
    print(missing_df['true_relevance'].value_counts().sort_index())
    print(f"仅针对真正相关视频的召回率: {true_recall:.2%}" if not np.isnan(true_recall) else "无 true_relevance>=3 样本，无法计算")

    # --- NDCG@20：左连接 + 固定 k=20，跟其余脚本统一算法，可直接放进同一张对比表 ---
    scored = eval_df.merge(ground_truth, on='id', how='left')
    scored['true_relevance'] = scored['true_relevance'].fillna(0)
    scored = scored.sort_values('bm25_score', ascending=False).reset_index(drop=True)

    true_rel = scored['true_relevance'].to_numpy().reshape(1, -1)
    pred_score = scored['bm25_score'].to_numpy().reshape(1, -1)
    k_actual = min(K, len(scored))
    score = ndcg_score(true_rel, pred_score, k=k_actual)

    n_judged_in_pool = eval_df['id'].isin(ground_truth['id']).sum()
    print(f"候选池大小: {len(eval_df)}   人工标注命中数: {n_judged_in_pool}/{len(ground_truth)}   NDCG@{k_actual} = {score:.4f}")
    print(scored[['combined_text', 'url', 'bm25_score', 'true_relevance']].head(20).to_string())
    print()

    eval_summary.append({
        "query": query_text, "method": "BM25",
        "candidate_pool_size": len(eval_df), "n_judged_matched": int(n_judged_in_pool),
        "n_judged_total": len(ground_truth), f"ndcg@{K}": round(score, 4),
        "true_recall(relevant>=3)": round(true_recall, 4) if not np.isnan(true_recall) else None,
    })

print("=== BM25 三个 query 汇总 ===")
print(pd.DataFrame(eval_summary).to_string(index=False))
