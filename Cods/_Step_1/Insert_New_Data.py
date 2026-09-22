import pandas as pd 
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils.exceptions import IllegalCharacterError

file_path = "D:/Users/User/Desktop/TikTok_Portfolio/Datasets/test.parquet"

df = pd.read_parquet(file_path) 

new_input = {
            'desc' : 'I cant buy figures because Im broke😾',
            'collect_count' : 5279,
            'comment_count' : 224,
            'digg_count'    : 22300,
            'play_count'    : 800000,
            'share_count'   : 2153,
            'duration'      : 7,
            #'city'          : null,
            'challenges'    : '["#rem","#rezero","#cosplay","#リゼロ","#レム"]',
            'url'    : 'https://www.tiktok.com/@aria_amamiya_nya/video/7674956418033765639'
            }

# 3. 将输入数据转换为单行的 DataFrame
df_new = pd.DataFrame([new_input])

# 4. 【修改这里】先合并两个 DataFrame
# 此时 Pandas 会自动把新行里缺失的列（比如 'id'）自动补上空值（NaN/Null）
df_combined = pd.concat([df,df_new],ignore_index=True)

# 5. 【修改这里】合并后再统一强制转换数据类型
# 这样不仅能确保新输入的数字类型正确，而且不会因为缺失列名而报错了
df_combined = df_combined.astype(df.dtypes.to_dict(), errors='ignore')

# 6. 保存回原本的文件路径
df_combined.to_parquet(file_path, index=True)
