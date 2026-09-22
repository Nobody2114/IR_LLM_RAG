#Dense_VSM 比 VSM拥有更多的customize，但是有可能精准度会降低.
import os
import pickle
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login
from sklearn.metrics import ndcg_score

df = pd.read_parquet("D:/Users/User/Desktop/TikTok_Portfolio/Datasets/test.parquet")
login(token="#")

# 【修复】跟 Twist.py 一样，统一从 url 反推精确 id，避免 float64 精度丢失
# 导致后面 merge ground truth 时对不上号。
_url_extracted_id = pd.to_numeric(df['url'].str.extract(r'/video/(\d+)')[0], errors='coerce')
_original_id = pd.to_numeric(df['id'], errors='coerce')
df['id'] = _url_extracted_id.fillna(_original_id).astype('Int64')

# 1. 准备文本数据
df["combined_text"] = df["desc"].fillna("") + " " + df["challenges"].fillna("")
corpus = df["combined_text"].astype(str).tolist()

# 2. 定义你想检索的查询
queries = [
    "Rem cosplay",
    "Genshin Cosplay",
    "Logo Design",
]

# 每个 query 对应它在字面匹配里要检查的独立关键词（Boolean/hardcode 方法
# 不像 Twist.py 那样会自动把 query 拆词加 IDF 权重，这里手动列出来，
# 跟原本硬编码 "Rem"/"cosplay" 的做法保持一致的风格，只是泛化成每个 query 各自一套）
QUERY_KEYWORDS = {
    "Rem cosplay": ["Rem", "cosplay"],
    "Genshin Cosplay": ["Genshin", "cosplay"],
    "Logo Design": ["Logo", "Design"],
}

# 3. 加载预训练的 Sentence-Transformers 模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 4. 直接加载已存在的 embedding 文件，跳过重复编码
embedding_file = "D:/Users/User/Desktop/TikTok_Portfolio/Cache/corpus_embeddings_cache.pkl"

if os.path.exists(embedding_file):
    print("Loading corpus embeddings from local cache...")
    with open(embedding_file, "rb") as f:
        corpus_embeddings = pickle.load(f)
else:
    print("Encoding corpus (first time)...")
    corpus_embeddings = model.encode(corpus, show_progress_bar=True)
    with open(embedding_file, "wb") as f:
        pickle.dump(corpus_embeddings, f)
    print("Embeddings saved to cache.")

def has_word(text, word):
    pattern = r'(?:#|\b)' + re.escape(word) + r'\b'
    return bool(re.search(pattern, text, re.IGNORECASE))

# 【新增】读取 Inverted Index 缓存，用来加速下面的关键词匹配
index_path = "D:/Users/User/Desktop/TikTok_Portfolio/Cache/inverted_index_cache.pkl"
with open(index_path, "rb") as f:
    inverted_index = pickle.load(f)

# === 【修复①】真正按 query 循环，而不是只处理第一个 ===
all_query_dfs = []

for q_id, query_text in enumerate(queries):
    keywords = [w.lower() for w in QUERY_KEYWORDS[query_text]]

    # 5. 这个 query 自己的相似度（不再用 [0] 固定切第一行）——
    # 这一步没办法用 Inverted Index 加速，Dense Vector 语义相似度天生
    # 就得对全部语料算一遍，见上一轮的说明。
    query_embedding = model.encode(query_text)
    similarity_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()

    temp_df = df.copy()
    temp_df["query_id"] = q_id
    temp_df["similarity_score"] = similarity_scores

    # 【核心改动】关键词匹配不再用 df["combined_text"].apply(正则扫描每一行)，
    # 改成查 Inverted Index 拿到"包含这个词的视频 id 集合"，再用 isin()
    # 一次性向量化判断每一行有没有命中——这才是索引该发挥作用的地方，
    # 原本那种逐行跑正则的写法，本质上还是暴力扫描，只是换了个形式。
    # 注意：索引用的是简单分词（小写、只留字母数字），跟原本 has_word()
    # 的 \b 整词边界正则不完全等价，绝大多数情况结果一致，极少数含特殊
    # 符号的边界情况可能略有出入——这是用索引换速度的取舍，如实记录。
    match_flags = pd.DataFrame({
        w: temp_df['id'].isin(inverted_index.get(w, set()))
        for w in keywords
    })
    all_matched = match_flags.all(axis=1)   # 全部关键词都命中
    any_matched = match_flags.any(axis=1)   # 至少命中一个

    temp_df["boosted_score"] = temp_df["similarity_score"] + np.where(
        all_matched, 0.4, np.where(any_matched, 0.2, 0.0)
    )
    keyword_mask = any_matched

    vsm_df = temp_df[(temp_df["similarity_score"] > 0.3) & keyword_mask].copy()

    if len(vsm_df) == 0:
        print(f"⚠️ [{query_text}] 没有任何候选通过筛选（相似度>0.3 且命中关键词），跳过。")
        continue

    weights = {'collect_count': 2, 'comment_count': 2, 'digg_count': 1, 'play_count': 1, 'share_count': 3}
    features = ['collect_count', 'comment_count', 'digg_count', 'play_count', 'share_count']
    for feat in features:
        vsm_df[f'{feat}_log'] = np.log1p(vsm_df[feat])

    vsm_df['score'] = (
        weights['collect_count'] * vsm_df['collect_count_log'] +
        weights['comment_count'] * vsm_df['comment_count_log'] +
        weights['digg_count'] * vsm_df['digg_count_log'] +
        weights['play_count'] * vsm_df['play_count_log'] +
        weights['share_count'] * vsm_df['share_count_log']
    )

    vsm_df = vsm_df.sort_values(by=['boosted_score', 'score'], ascending=False).reset_index(drop=True)
    all_query_dfs.append(vsm_df)

master_df = pd.concat(all_query_dfs, ignore_index=True)

# === Ground truth 加载（跟 Twist.py 那份一致的逻辑） ===
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
        print(f"⚠️ {path} 里有 {len(bad_rows)} 行 true_relevance 不是纯数字，检查一下这几行：")
        print(bad_rows[['id', 'url']].to_string())
        gt = gt.dropna(subset=['true_relevance'])

    return gt[['id', 'true_relevance']]

# === 【修复③】评估：Dense_VSM_Hardcode 是纯规则打分，不训练模型，
# 直接用 boosted_score 排序去跟 ground truth 比，不需要 leave-one-query-out ===
K = 20
eval_summary = []

for query_text, gt_path in GROUND_TRUTH_FILES.items():
    if query_text not in queries:
        print(f"⚠️ 跳过 [{query_text}]：不在 queries 列表里。")
        continue
    if not os.path.exists(gt_path):
        print(f"⚠️ 跳过 [{query_text}]：找不到 ground truth 文件 {gt_path}")
        continue

    gt_query_id = queries.index(query_text)
    ground_truth = load_ground_truth(gt_path)

    if gt_query_id not in master_df['query_id'].unique():
        print(f"⚠️ 跳过 [{query_text}]：这个 query 在候选生成阶段没有产出任何候选。")
        continue

    eval_df = master_df[master_df['query_id'] == gt_query_id].copy()

    _n_before = len(eval_df)
    eval_df = eval_df.dropna(subset=['id'])
    _n_dropped = _n_before - len(eval_df)
    if _n_dropped > 0:
        print(f"⚠️ [{query_text}] 候选池里有 {_n_dropped} 条视频 id 缺失，评估阶段已跳过这些行。")
    eval_df['id'] = eval_df['id'].astype('int64')

    # --- 漏召回诊断：ground truth 里，有多少条没进到这个方法的候选池 ---
    missing_ids = set(ground_truth['id']) - set(eval_df['id'])
    missing_df = ground_truth[ground_truth['id'].isin(missing_ids)]

    relevant_gt = ground_truth[ground_truth['true_relevance'] >= 3]
    relevant_missing = missing_df[missing_df['true_relevance'] >= 3]
    true_recall = (1 - len(relevant_missing) / len(relevant_gt)) if len(relevant_gt) > 0 else float('nan')

    print(f"=== [{query_text}] Dense_VSM_Hardcode 召回诊断 ===")
    print(f"漏召回总数: {len(missing_df)}")
    print(f"其中 true_relevance = 0 (本来就不相关): {(missing_df['true_relevance'] == 0).sum()}")
    print(f"其中 true_relevance >= 3 (人工判定明确相关): {(missing_df['true_relevance'] >= 3).sum()}")
    print(missing_df['true_relevance'].value_counts().sort_index())
    print(f"仅针对真正相关视频的召回率: {true_recall:.2%}" if not np.isnan(true_recall) else "该 query 没有 true_relevance>=3 的 ground truth，无法算召回率")

    # --- NDCG@K：左连接，没被标注过的候选一律视为不相关（true_relevance=0），
    # 这是标准做法——ground truth 只标了"我认为相关"的那些，没提到的不代表
    # 真的不相关，但在计算某个固定 k 的 NDCG 时，这是唯一自洽的处理方式 ---
    scored = eval_df.merge(ground_truth, on='id', how='left')
    scored['true_relevance'] = scored['true_relevance'].fillna(0)
    scored = scored.sort_values('boosted_score', ascending=False).reset_index(drop=True)

    true_rel = scored['true_relevance'].to_numpy().reshape(1, -1)
    pred_score = scored['boosted_score'].to_numpy().reshape(1, -1)
    k_actual = min(K, len(scored))
    score = ndcg_score(true_rel, pred_score, k=k_actual)

    n_judged_in_pool = eval_df['id'].isin(ground_truth['id']).sum()
    print(f"候选池大小: {len(eval_df)}   人工标注命中数: {n_judged_in_pool}/{len(ground_truth)}   NDCG@{k_actual} = {score:.4f}\n")
    print(scored[['combined_text', 'url', 'true_relevance']].head(20).to_string())

    eval_summary.append({
        "query": query_text,
        "method": "Dense_VSM_Hardcode",
        "candidate_pool_size": len(eval_df),
        "n_judged_matched": int(n_judged_in_pool),
        "n_judged_total": len(ground_truth),
        f"ndcg@{K}": round(score, 4),
        "true_recall(relevant>=3)": round(true_recall, 4) if not np.isnan(true_recall) else None,
    })

print("=== Dense_VSM_Hardcode 三个 query 汇总 ===")
print(pd.DataFrame(eval_summary).to_string(index=False))
