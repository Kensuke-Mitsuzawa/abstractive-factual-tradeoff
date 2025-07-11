# %%
import json
from pathlib import Path
from summary_abstractive import visualization_header

# %%
PATH_DATASET_CNN = Path("/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/datasets/constraints_fact_v1.0/cnn_dailymail/collect.json")
assert PATH_DATASET_CNN.exists()

# %%
with PATH_DATASET_CNN.open() as f:
    seq_dataset = [json.loads(_line) for _line in f.readlines()]

# %%
# I want to extract fields of "id", "document_full", "annotator_votes", "abstractiveness_constraint"
seq_dataset_light = []
for _obj in seq_dataset:
    sum_annotator_votes = sum(_obj["annotator_votes"])
    seq_dataset_light.append({
        "document_id": _obj["document_id"], 
        "document_full": _obj["document_full"],
        "sum_annotator_votes": int(sum_annotator_votes), 
        "annotator_votes": _obj["annotator_votes"], 
        "abstractiveness_constraint": _obj["abstractiveness_constraint"]
    })
# end for

# %%
path_dir_stats_out = Path("./constraints_fact_v1.0-cnn_dailymail")
path_dir_stats_out.mkdir(parents=True, exist_ok=True)

# %%
# x: sum of annotator_votes [0 - 3]. y: freq]
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns
import pandas as pd

df_freq_votes = pd.DataFrame(seq_dataset_light)
df_freq_votes["sum_annotator_votes"] = df_freq_votes["sum_annotator_votes"].astype(int)

_f, _ax = plot.subplots()

_ax = sns.histplot(data=df_freq_votes, x="sum_annotator_votes", 
                   hue="abstractiveness_constraint", 
                   multiple="dodge",  # 👈 Request 2: Display hue groups side-by-side
                   discrete=True,     # 👈 Request 1: Treat x-axis as discrete integer categories
                   shrink=0.8,         # Optional: Narrows the bars to add space between groups                   
                   ax=_ax)

# _ax = _ax.legend(loc='center')
sns.move_legend(_ax, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.3), frameon=False, title=None)

_f.savefig((path_dir_stats_out / "histoplot.png").as_posix(), dpi=300, bbox_inches='tight')

# %%
df_agg = df_freq_votes.groupby(by=["abstractiveness_constraint", "sum_annotator_votes"]).agg({"sum_annotator_votes": "count"})
df_agg.columns = ["count_sum_annotator_votes"]
df_agg.sort_values(by="count_sum_annotator_votes", inplace=True, ascending=False)
df_agg.reset_index(inplace=True)
df_agg["ratio"] = df_agg["count_sum_annotator_votes"] / df_agg["count_sum_annotator_votes"].sum()
df_agg.to_csv(path_dir_stats_out / "stats.csv", sep=",", index=False)

# %%
len(df_freq_votes)

# %%
len(df_freq_votes[df_freq_votes["sum_annotator_votes"] == 0])

# %%


# %%


# %%



