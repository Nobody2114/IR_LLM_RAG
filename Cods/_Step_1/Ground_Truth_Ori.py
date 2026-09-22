import os
import re
import ast
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import wordninja
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login

# 1. 基础数据加载与登录
df = pd.read_parquet("D:/Users/User/Desktop/TikTok_Portfolio/Datasets/test.parquet")
login(token="#")

# 【修复⑧】部分行（比如手动 insert 进 parquet 的 Rem 视频）本身没有 id，
# 但有真实的 TikTok url，可以从 url 里 /video/ 后面的数字反推出精确 id。
# 优先信任 url 提取出来的数字：即使原本的 id 列存在，只要整列曾经混进过
# NaN，pandas 就会把整列强制转成 float64，19 位长整数在 float64 下精度
# 不够，后几位会被悄悄抹掉——所以原本"看起来有 id"的行也不一定可信。
# 只有 url 本身解析不出数字的极少数行，才退回去用原本的 id 列兜底。
# 最终统一存成 pandas 的可空整数类型 Int64（注意大写 I）——它既能表示
# NaN，又不会像 float64 那样丢失大整数的精度，两边都占了。
_url_extracted_id = pd.to_numeric(df['url'].str.extract(r'/video/(\d+)')[0], errors='coerce')
_original_id = pd.to_numeric(df['id'], errors='coerce')
df['id'] = _url_extracted_id.fillna(_original_id).astype('Int64')

_still_missing = df['id'].isna().sum()
if _still_missing > 0:
    print(f"⚠️ 有 {_still_missing} 行既没有可用 id，url 也解析不出数字，"
          f"这些行永远没办法跟任何 ground truth 的 id merge 上，建议检查一下这些行的来源。")

df["combined_text"] = df["desc"].fillna("") + " " + df["challenges"].fillna("")
df["challenges_str"] = df["challenges"].fillna("").astype(str)
corpus = df["combined_text"].astype(str).tolist()

queries = [
    "Rezero Cosplay",
    "Rem cosplay",
    "Funny Cat Videos",
    "Graphic Design",
    "Poster Concept",
    "Anime",
    "Nintendo Game",
    "Genshin Cosplay",   # 【新增】GenshinCos_Ground_Truth.xlsx 对应的 query
    "Logo Design",       # 【新增】Logo_Design_Ground_Truth.xlsx 对应的 query
]

# === 核心软特征 (Soft Features) 提取逻辑 ===
def has_word(text, word):
    pattern = r'(?:#|\b)' + re.escape(word) + r'\b'
    return bool(re.search(pattern, str(text), re.IGNORECASE))

# --- 【修复②】IDF 权重：让罕见词（如 "Rem"）比高频通用词（如 "cosplay"）更重要 ---
# 原来 match_ratio 是简单的 matched_count / len(words)，对 "Rem cosplay" 这种查询，
# 只匹配到 "cosplay"（语料库里的高频词）和只匹配到 "Rem"（罕见、更有区分度的词）
# 得到的 match_ratio 完全一样（都是 0.5）。这会导致候选池里 99%+ 的样本
# 在相关性特征上几乎无法区分，只能靠互动量（播放/点赞数）分高下。
_word_idf_cache = {}
_corpus_lower_cache = None
_N_DOCS = len(corpus)

def get_word_idf(word):
    """语料库内该词的逆文档频率：越少见的词权重越高。"""
    global _corpus_lower_cache
    word = word.lower()
    if word in _word_idf_cache:
        return _word_idf_cache[word]
    if _corpus_lower_cache is None:
        _corpus_lower_cache = [str(t).lower() for t in corpus]
    doc_freq = sum(1 for t in _corpus_lower_cache if has_word(t, word))
    idf = np.log((_N_DOCS + 1) / (doc_freq + 1)) + 1
    _word_idf_cache[word] = idf
    return idf

# --- 【修复⑤a】hashtag 拼接词识别：#remcosplay / #cosplaygirls 这类
# 把多个词粘在一起的 hashtag，无法用 \bword\b 整词边界匹配识别出内部的词。
# 需要把 challenges 字段拆成单个 hashtag token，再对每个 token 分别做子串匹配
# （只在同一个 token 内部做子串匹配，不跨 token，避免"两个不相关 hashtag
# 拼起来偶然形成误判"的风险）。
def parse_hashtag_tokens(challenges_str):
    """把形如 '[\"remcosplay\",\"fyp\",...]' 的字符串解析成小写 token 列表；
    解析失败就退回把整个字符串当一个 token（至少不会比原来更差）。"""
    try:
        tokens = ast.literal_eval(str(challenges_str))
        if isinstance(tokens, (list, tuple)):
            return [str(t).lower() for t in tokens]
    except (ValueError, SyntaxError):
        pass
    return [str(challenges_str).lower()]

# wordninja.split 对每个独立 hashtag token 是纯函数（同样的字符串结果一定一样），
# 但语料库里 hashtag 重复率很高（"#fyp"、"#cosplay" 会在成千上万条视频里反复出现），
# 每次都重新切词非常浪费。用一个跨行共享的全局缓存，切一次记住结果。
_hashtag_segment_cache = {}

def word_in_hashtags(hashtag_tokens, word, segmented_cache):
    """
    【修复⑥】不再用裸的子串检测（word in tok），因为这会把 "remake" 里的
    "rem" 也算作命中——"remake" 是一个完整单词，跟 "rem"+"cosplay" 这种
    真正由两个查询词拼接成的 hashtag（如 "remcosplay"）性质完全不同，
    但裸子串检测分不出这个区别，导致 "Resident Evil remake cosplay" 这种
    完全无关的内容被误判为命中了 "Rem"。

    改用 wordninja 做词频驱动的切词：把每个 hashtag token 切成它"最可能"
    对应的单词序列，再检查 query word 是否是切分结果里的某一个独立词。
    "remcosplay" 会被切成 ["rem", "cosplay"]（两个查询词拼接，符合预期）；
    "remake" 作为一个高频真实单词，切词时会保持整体，不会被拆出 "rem"。
    """
    word = word.lower()
    for tok in hashtag_tokens:
        if tok not in segmented_cache:
            segmented_cache[tok] = [w.lower() for w in wordninja.split(tok)]
        if word in segmented_cache[tok]:
            return True
    return False

def extract_soft_features(row, query_text):
    """
    提取细粒度的匹配软特征，为 LTR 模型提供强语义信号。
    【修复②】match_ratio 改为 IDF 加权。
    【修复⑤a→⑥】hashtag 命中判定不再用裸子串匹配，改用 wordninja 词切分后
    的精确词匹配，避免 "remake" 这类整体单词被误判为包含独立的查询词 "rem"。
    【修复⑤b】未完全命中的样本，match_ratio 额外打折。
    """
    text = str(row["combined_text"]).lower()
    hashtag_tokens = parse_hashtag_tokens(row["challenges_str"])
    query_clean = query_text.lower().strip()
    words = query_clean.split()

    if not words:
        return 0, 0.0, 0, 0

    # segmented_cache 按行局部缓存即可；真正的性能优化见下方的全局缓存说明
    segmented_cache = _hashtag_segment_cache

    # 1. 命中判定：正文整词匹配 OR 出现在某个 hashtag token 切词后的结果里
    matches = [has_word(text, w) or word_in_hashtags(hashtag_tokens, w, segmented_cache) for w in words]
    matched_count = sum(matches)
    exact_match = 1 if matched_count == len(words) else 0

    word_idfs = [get_word_idf(w) for w in words]
    total_idf = sum(word_idfs) or 1e-6
    matched_idf = sum(idf for m, idf in zip(matches, word_idfs) if m)
    raw_match_ratio = matched_idf / total_idf

    # 未完全命中时打 0.5 折，降低"只蹭到一个词"的匹配的分量
    match_ratio = raw_match_ratio if exact_match == 1 else raw_match_ratio * 0.5

    # 2. phrase_match: 查询词作为完整短语连续出现（正文，或去掉空格后出现在某个 hashtag token 里）
    query_no_space = query_clean.replace(" ", "")
    phrase_match = 1 if (
        query_clean in text or any(query_no_space in tok for tok in hashtag_tokens)
    ) else 0

    # 3. tag_match: 任一 query word 是否出现在任一 hashtag token 切词后的结果里
    tag_match = 1 if any(word_in_hashtags(hashtag_tokens, w, segmented_cache) for w in words) else 0

    return exact_match, match_ratio, phrase_match, tag_match

# 2. 加载语义向量模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embedding_file = "D:/Users/User/Desktop/TikTok_Portfolio/Cache/corpus_embeddings_cache.pkl"

if os.path.exists(embedding_file):
    with open(embedding_file, "rb") as f:
        corpus_embeddings = pickle.load(f)
else:
    corpus_embeddings = model.encode(corpus, show_progress_bar=True)
    with open(embedding_file, "wb") as f:
        pickle.dump(corpus_embeddings, f)

all_query_dfs = []

# === 3. 遍历 Query，构建候选集与软伪标签 ===
for q_id, query_text in enumerate(queries):

    # 向量计算
    query_embedding = model.encode(query_text)
    similarity_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()

    temp_df = df.copy()
    temp_df["query_id"] = q_id
    temp_df["similarity_score"] = similarity_scores

    # 应用软特征提取
    features_df = temp_df.apply(lambda r: extract_soft_features(r, query_text), axis=1, result_type='expand')
    temp_df[["exact_match", "match_ratio", "phrase_match", "tag_match"]] = features_df

    # 【宽召回条件】保持不变：向量有相似度且至少命中一点词汇就进入粗筛。
    # 召回池仍然要宽，这样每个 query 才有足够多的候选和足够的组内样本量
    # 供 LightGBM 学习对比；真正的过滤发生在下面的 engagement_score 硬门控里。
    sub_df = temp_df[(temp_df["similarity_score"] > 0.15) & (temp_df["match_ratio"] > 0)].copy()

    if len(sub_df) > 0:
        sub_df = sub_df.nlargest(min(1000, len(sub_df)), "similarity_score")

        if len(sub_df) > 0:
            # 基础互动量 (Log 缩放)
            sub_df['collect_log'] = np.log1p(sub_df['collect_count'])
            sub_df['comment_log'] = np.log1p(sub_df['comment_count'])
            sub_df['digg_log'] = np.log1p(sub_df['digg_count'])
            sub_df['play_log'] = np.log1p(sub_df['play_count'])
            sub_df['share_log'] = np.log1p(sub_df['share_count'])

            raw_engagement = (
                2 * sub_df['collect_log'] +
                2 * sub_df['comment_log'] +
                1 * sub_df['digg_log'] +
                3 * sub_df['share_log'] +
                1 * sub_df['play_log']
            )

            # 互动量归一化到 (0 ~ 1)
            e_min, e_max = raw_engagement.min(), raw_engagement.max()
            norm_engagement = (raw_engagement - e_min) / (e_max - e_min + 1e-6)

            # === 【核心伪标签算法】：软结合 (Soft Labeling Function) ===
            # 综合相关性得分 (0.0 ~ 1.0)：结合 IDF 加权比例、精准短语、Tag 和向量得分
            relevance_composite = (
                0.40 * sub_df['match_ratio'] +
                0.25 * sub_df['exact_match'] +
                0.20 * sub_df['phrase_match'] +
                0.15 * sub_df['similarity_score']
            )

            # 【修复③】硬门控代替"软天花板"：
            # 原来的乘积公式本意是让低相关性压低高互动，但当 relevance_composite
            # 在大部分候选里几乎恒定时，天花板形同虚设，engagement 会线性主导得分。
            # 现在明确规定：只有 exact_match==1（查询里的所有词都真正命中）的样本
            # 才允许互动量参与打分；没有完全命中的样本，无论多热门，
            # engagement_score 都被强制压到 relevance_composite 的一个小零头，
            # 保证它们不可能挤进高分段、污染 relevance_label。
            is_true_match = (sub_df['exact_match'] == 1)
            gated_score = (0.7 * relevance_composite + 0.3 * norm_engagement) * relevance_composite
            sub_df['engagement_score'] = np.where(
                is_true_match,
                gated_score,
                relevance_composite * 0.1
            )

            # 【修复④】relevance_label 不能再用一次性的 qcut 等频分箱来算。
            # 原因：is_true_match 的样本极少（比如 1000 条里只有 3 条），
            # qcut 是按百分位数切的，不管这 3 条 engagement_score 比其余样本
            # 高出多少倍，只要它们恰好落在前 20% 的位置，就会跟另外一大票
            # "沾边但不相关"的样本共享同一个 label=4。而 LightGBM 的
            # lambdarank/NDCG 损失对同一 label 的样本之间不产生梯度——
            # 模型完全没有被要求把真正匹配的排到这些凑数样本前面，
            # 组内相对顺序就变成了看运气。
            #
            # 修复方式：真正完全匹配（exact_match==1）的样本，无条件拿最高档
            # 标签，跟其余样本的分箱彻底分开；其余样本再单独按 engagement_score
            # 分箱，占据较低的档位。这样保证"真正相关"永远比"沾边但不相关"
            # 的标签更高，训练时才有梯度信号去把它们排到前面。
            TOP_LABEL = 4
            n_lower_bins = TOP_LABEL  # 剩余样本占用 0 ~ TOP_LABEL-1 档

            sub_df['relevance_label'] = 0
            true_mask = sub_df['exact_match'] == 1
            other_mask = ~true_mask

            if other_mask.sum() > 0:
                sub_df.loc[other_mask, 'relevance_label'] = pd.qcut(
                    sub_df.loc[other_mask, 'engagement_score'],
                    q=min(n_lower_bins, sub_df.loc[other_mask, 'engagement_score'].nunique()),
                    labels=False,
                    duplicates='drop'
                )

            # 真正完全匹配的样本，不管多少条、engagement 高低，一律拿最高档标签。
            sub_df.loc[true_mask, 'relevance_label'] = TOP_LABEL

            sub_df['relevance_label'] = sub_df['relevance_label'].astype(int)
            all_query_dfs.append(sub_df)

# 合并与排序
master_df = pd.concat(all_query_dfs, ignore_index=True)
master_df = master_df.sort_values(by="query_id").reset_index(drop=True)

# === 4. LTR 模型基础特征列 ===
feature_cols = [
    'similarity_score',
    'exact_match',      # 软特征：完全命中
    'match_ratio',      # 软特征：命中比例（IDF 加权）
    'phrase_match',     # 软特征：连续短语命中
    'tag_match',        # 软特征：Hashtag 匹配
    'collect_log',
    'comment_log',
    'digg_log',
    'play_log',
    'share_log'
]

# === 5. 用 3 份人工 ground truth，逐个做 leave-one-query-out 泛化测试 ===
#
# 每份 ground truth 对应一个 query：训练时把这个 query 整个从训练集里剔除，
# 只用其余 query 训练模型，再让模型去预测它完全没见过的这个 query 的候选池，
# 跟你的人工判断对比算 NDCG。这样才是真正没有 leakage 的评估——
# 如果不排除，等于在模型训练时见过的数据上打分，分数会虚高。
GROUND_TRUTH_FILES = {
    "Rem cosplay":     "D:/Users/User/Desktop/TikTok_Portfolio/Ground_Truth/Rem_Ground_truth.xlsx",
    "Genshin Cosplay": "D:/Users/User/Desktop/TikTok_Portfolio/Ground_Truth/GenshinCos_Ground_Truth.xlsx",
    "Logo Design":     "D:/Users/User/Desktop/TikTok_Portfolio/Ground_Truth/LogoDesign_Ground_Truth.xlsx",
}

def load_ground_truth(path):
    """
    读取人工标注的 ground truth excel。
    不信任文件里的 id 列——19 位长整数一旦在 excel 里跟任何 NaN 混在同一列，
    pandas/excel 就会把整列转成 float64，导致精度丢失、后几位数字变成乱码。
    统一改用 url 里 /video/ 后面的数字重新提取一份精确 id 来做 merge key。
    标签列名兼容 "relevance_label"（Rem 那份）和 "relevant"（Genshin 那份）。
    """
    gt = pd.read_excel(path)
    gt['id'] = gt['url'].str.extract(r'/video/(\d+)').astype('int64')
    label_col = 'relevance_label' if 'relevance_label' in gt.columns else 'relevant'
    gt = gt.rename(columns={label_col: 'true_relevance'})
    return gt[['id', 'true_relevance']]

from sklearn.metrics import ndcg_score

eval_summary = []

for query_text, gt_path in GROUND_TRUTH_FILES.items():
    if query_text not in queries:
        print(f"⚠️ 跳过 [{query_text}]：不在 queries 列表里，加进去、重新跑一遍第 3 步候选生成后再评估。")
        continue

    if not os.path.exists(gt_path):
        print(f"⚠️ 跳过 [{query_text}]：找不到 ground truth 文件 {gt_path}")
        continue

    gt_query_id = queries.index(query_text)
    ground_truth = load_ground_truth(gt_path)

    if gt_query_id not in master_df['query_id'].unique():
        print(f"⚠️ 跳过 [{query_text}]：这个 query 在候选生成阶段（第3步）没有产出任何候选，"
              f"检查一下召回条件（similarity_score>0.15 & match_ratio>0）是不是把它全部筛掉了。")
        continue

    # leave-one-query-out：这个 query 完全不参与训练
    train_df = master_df[master_df['query_id'] != gt_query_id]
    eval_df = master_df[master_df['query_id'] == gt_query_id].copy()

    # 【修复⑧续】id 已经在源头（第1步）统一修好了，这里理论上不该再有 NaN；
    # 万一真的还有极少数 id/url 都解析不出来的行，与其让 astype 直接崩溃，
    # 不如明确丢掉这些行并打印出来——它们本来就永远不可能跟任何 ground
    # truth 的 id merge 上，留着也没用，但要让你知道丢了几条、别悄悄发生。
    _n_before = len(eval_df)
    eval_df = eval_df.dropna(subset=['id'])
    _n_dropped = _n_before - len(eval_df)
    if _n_dropped > 0:
        print(f"⚠️ [{query_text}] 候选池里有 {_n_dropped} 条视频 id 缺失（无法从 url 补齐），"
              f"评估阶段已跳过这些行。")
    eval_df['id'] = eval_df['id'].astype('int64')

    X_train = train_df[feature_cols]
    y_train = train_df['relevance_label'].astype(int)
    group_train = train_df.groupby('query_id').size().to_list()

    ranker_holdout = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        learning_rate=0.1, n_estimators=100, random_state=42
    )
    ranker_holdout.fit(X_train, y_train, group=group_train)
    eval_df['ltr_score'] = ranker_holdout.predict(eval_df[feature_cols])

    # 诊断：ground truth 里有多少条根本没进到 Twist.py 自己的候选池——
    # 这些是召回阶段漏掉的，不是排序排错了，两者要分开看，不能都算进排序准确率里。
    missing_ids = set(ground_truth['id']) - set(eval_df['id'])
    if missing_ids:
        print(f"⚠️ [{query_text}] 有 {len(missing_ids)} 条人工标注视频没进候选池"
              f"（大概率是召回阶段漏掉，不是排序问题）: {sorted(missing_ids)}")

    judged = eval_df.merge(ground_truth, on='id', how='inner')

    if len(judged) == 0:
        print(f"⚠️ [{query_text}] ground truth 一条都没匹配上候选池，跳过 NDCG 计算。\n")
        continue

    k = len(judged)
    true_rel = judged['true_relevance'].to_numpy().reshape(1, -1)
    pred_score = judged['ltr_score'].to_numpy().reshape(1, -1)
    score = ndcg_score(true_rel, pred_score, k=20)

    print(f"=== [{query_text}] 泛化测试（模型训练时完全没见过这个 query）===")
    print(f"候选池大小: {len(eval_df)}   人工标注命中数: {len(judged)}/{len(ground_truth)}   NDCG@{k} = {score:.4f}")
    print(judged.sort_values('ltr_score', ascending=False)[
        ['combined_text', 'ltr_score', 'true_relevance']
    ].head(10).to_string())
    print()

    eval_summary.append({
        "query": query_text,
        "candidate_pool_size": len(eval_df),
        "n_judged_matched": len(judged),
        "n_judged_total": len(ground_truth),
        "ndcg": round(score, 4),
    })

print("=== 三个 query 的泛化测试汇总 ===")
print(pd.DataFrame(eval_summary).to_string(index=False))

# === 6. 用全部 query（含刚才 hold-out 的）训练一个最终模型，用于实际展示排序结果 ===
# 上面几轮训练是为了「诚实评估泛化能力」而故意排除某个 query；
# 评估完之后，实际要展示/使用的排序结果，理应用全部数据训练的模型来产出，
# 不需要再故意藏着已知信息。
X_full = master_df[feature_cols]
y_full = master_df['relevance_label'].astype(int)
group_full = master_df.groupby('query_id').size().to_list()

ranker = lgb.LGBMRanker(
    objective="lambdarank", metric="ndcg",
    learning_rate=0.1, n_estimators=100, random_state=42
)
ranker.fit(X_full, y_full, group=group_full)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": ranker.feature_importances_
}).sort_values("importance", ascending=False)
print("\n=== 特征重要性（最终全量模型）===")
print(importance_df.to_string(index=False))

master_df['ltr_score'] = ranker.predict(master_df[feature_cols])
final_result = master_df.sort_values(by=['query_id', 'ltr_score'], ascending=[True, False])

#final_result.to_excel('D:/Users/User/Desktop/TikTok_Portfolio/Ground_Truth_T3.xlsx', index=False)
