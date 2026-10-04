import pandas as pd
import torch
from torch_geometric.data import Data

ratings = pd.read_csv(
    "data/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Keep positive interactions
ratings = ratings[ratings["rating"] >= 4].copy()

# Map original IDs → consecutive node IDs
user_ids = ratings["user_id"].unique()
movie_ids = ratings["movie_id"].unique()

user_map = {uid: i for i, uid in enumerate(user_ids)}
movie_map = {
    mid: i + len(user_ids)
    for i, mid in enumerate(movie_ids)
}

# Build edges
src = []
dst = []

for row in ratings.itertuples():
    user_node = user_map[row.user_id]
    movie_node = movie_map[row.movie_id]

    # User → Movie
    src.append(user_node)
    dst.append(movie_node)

    # Movie → User
    src.append(movie_node)
    dst.append(user_node)

edge_index = torch.tensor([src, dst], dtype=torch.long)

num_nodes = len(user_ids) + len(movie_ids)

graph = Data(
    edge_index=edge_index,
    num_nodes=num_nodes
)

print("Users:", len(user_ids))
print("Movies:", len(movie_ids))
print("Nodes:", graph.num_nodes)
print("Edges:", graph.edge_index.shape[1])
print("Edge index", graph.edge_index)

torch.save(
    {
        "graph": graph,
        "user_map": user_map,
        "movie_map": movie_map,
    },
    "data/graph.pt"
)