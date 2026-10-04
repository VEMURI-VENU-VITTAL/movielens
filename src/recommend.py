import torch
import faiss
import numpy as np


# -----------------------------
# Load embeddings
# -----------------------------

data = torch.load(
    "data/embeddings.pt",
    weights_only=False
)

embeddings = data["embeddings"]
user_map = data["user_map"]
movie_map = data["movie_map"]


# Reverse movie map:
# internal movie node ID -> original MovieLens movie ID
reverse_movie_map = {
    node_id: movie_id
    for movie_id, node_id in movie_map.items()
}


# -----------------------------
# Movie embeddings
# -----------------------------

movie_nodes = list(movie_map.values())

movie_embeddings = (
    embeddings[movie_nodes]
    .detach()
    .cpu()
    .numpy()
    .astype("float32")
)


# Normalize embeddings
faiss.normalize_L2(movie_embeddings)


# -----------------------------
# Build FAISS index
# -----------------------------

dimension = movie_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(movie_embeddings)


# -----------------------------
# Recommendation function
# -----------------------------

def recommend_movies(original_user_id, k=10):

    if original_user_id not in user_map:
        print("User not found.")
        return

    user_node = user_map[original_user_id]

    user_embedding = (
        embeddings[user_node]
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
        .reshape(1, -1)
    )

    faiss.normalize_L2(user_embedding)

    scores, indices = index.search(
        user_embedding,
        k
    )

    print(f"\nRecommendations for User {original_user_id}")
    print("-" * 40)

    for rank, (idx, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1
    ):

        movie_node = movie_nodes[idx]

        original_movie_id = reverse_movie_map[movie_node]

        print(
            f"{rank}. Movie ID: {original_movie_id} "
            f"| similarity: {score:.4f}"
        )


# -----------------------------
# Example
# -----------------------------

recommend_movies(
    original_user_id=1,
    k=10
)