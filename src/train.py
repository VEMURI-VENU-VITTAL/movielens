import torch
import torch.nn.functional as F
from model import GraphSAGE


# Load graph
data = torch.load("data/graph.pt", weights_only=False)

graph = data["graph"]
user_map = data["user_map"]
movie_map = data["movie_map"]

num_nodes = graph.num_nodes

# Model
model = GraphSAGE(
    num_nodes=num_nodes,
    hidden_dim=64,
    embedding_dim=32
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# Get user -> movie edges only
edges = graph.edge_index

user_movie_edges = []

for i in range(edges.shape[1]):
    src = edges[0, i].item()
    dst = edges[1, i].item()

    # User nodes are at the beginning
    if src in user_map.values() and dst in movie_map.values():
        user_movie_edges.append((src, dst))

print("Positive edges:", len(user_movie_edges))


for epoch in range(20):

    model.train()

    embeddings = model(graph.edge_index)

    # Positive samples
    pos_users = torch.tensor(
        [x[0] for x in user_movie_edges],
        dtype=torch.long
    )

    pos_movies = torch.tensor(
        [x[1] for x in user_movie_edges],
        dtype=torch.long
    )

    # Random negative movies
    neg_movies = torch.randint(
        min(movie_map.values()),
        max(movie_map.values()) + 1,
        (len(pos_movies),)
    )

    # Positive similarity
    pos_score = (
        embeddings[pos_users] *
        embeddings[pos_movies]
    ).sum(dim=1)

    # Negative similarity
    neg_score = (
        embeddings[pos_users] *
        embeddings[neg_movies]
    ).sum(dim=1)

    # BPR-style loss
    loss = -F.logsigmoid(
        pos_score - neg_score
    ).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(
            f"Epoch {epoch:02d} | Loss: {loss.item():.4f}"
        )


# Save embeddings
model.eval()

with torch.no_grad():
    embeddings = model(graph.edge_index)

torch.save(
    {
        "embeddings": embeddings,
        "user_map": user_map,
        "movie_map": movie_map,
    },
    "data/embeddings.pt"
)

print("Saved embeddings to data/embeddings.pt")