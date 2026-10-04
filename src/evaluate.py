import torch
import numpy as np
import math
import random


# -----------------------------
# Load graph
# -----------------------------

data = torch.load(
    "data/graph.pt",
    weights_only=False
)

graph = data["graph"]
user_map = data["user_map"]
movie_map = data["movie_map"]


# -----------------------------
# Load trained embeddings
# -----------------------------

emb_data = torch.load(
    "data/embeddings.pt",
    weights_only=False
)

embeddings = emb_data["embeddings"]


# -----------------------------
# Separate users and movies
# -----------------------------

user_nodes = set(user_map.values())
movie_nodes = set(movie_map.values())


# -----------------------------
# Build user -> movies mapping
# -----------------------------

user_movies = {}

edges = graph.edge_index

for i in range(edges.shape[1]):

    user = edges[0, i].item()
    movie = edges[1, i].item()

    if user in user_nodes and movie in movie_nodes:

        if user not in user_movies:
            user_movies[user] = []

        user_movies[user].append(movie)


# -----------------------------
# Create test set
# -----------------------------

test_pairs = []

for user, movies in user_movies.items():

    if len(movies) < 2:
        continue

    test_movie = random.choice(movies)

    test_pairs.append(
        (user, test_movie)
    )


# -----------------------------
# Movie embeddings
# -----------------------------

movie_node_list = list(movie_nodes)

movie_embeddings = (
    embeddings[movie_node_list]
    .detach()
    .cpu()
)


# -----------------------------
# Metrics
# -----------------------------

def recall_at_k(recommended, actual):

    return 1.0 if actual in recommended else 0.0


def ndcg_at_k(recommended, actual):

    if actual not in recommended:
        return 0.0

    rank = recommended.index(actual)

    return 1.0 / math.log2(rank + 2)


# -----------------------------
# Evaluate
# -----------------------------

K = 20

recall_scores = []
ndcg_scores = []


for user, test_movie in test_pairs:

    user_embedding = embeddings[user]

    scores = torch.matmul(
        movie_embeddings,
        user_embedding
    )

    top_indices = torch.topk(
        scores,
        K
    ).indices

    recommendations = [
        movie_node_list[i]
        for i in top_indices.tolist()
    ]

    recall_scores.append(
        recall_at_k(
            recommendations,
            test_movie
        )
    )

    ndcg_scores.append(
        ndcg_at_k(
            recommendations,
            test_movie
        )
    )


print("\nGraphSAGE Results")
print("-----------------------------")

print(
    f"Users evaluated: {len(test_pairs)}"
)

print(
    f"Recall@20: {np.mean(recall_scores):.4f}"
)

print(
    f"NDCG@20:   {np.mean(ndcg_scores):.4f}"
)