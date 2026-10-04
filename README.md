# GraphSAGE Movie Recommender

A graph-based movie recommendation system inspired by Pinterest's **PinSage** architecture. The project represents users and movies as a graph, learns node embeddings using **GraphSAGE**, and uses **FAISS** for fast nearest-neighbor candidate retrieval.

## Architecture


                 MovieLens-1M
                      |
                      v
              User-Movie Graph
                      |
                      v
                 GraphSAGE
                      |
                      v
           User / Movie Embeddings
                      |
                      v
                   FAISS
                      |
                      v
             Top-K Recommendations
                      |
                      v
             Recall@20 / NDCG@20
             

## Core Idea

Users and movies are represented as nodes in a graph.

An interaction between a user and a movie creates an edge:


User ───── Movie


GraphSAGE uses these connections to aggregate information from neighboring nodes and learn a vector representation for every user and movie.

For example:


User
 |
 +── Movie A
 +── Movie B
 +── Movie C

The user's embedding is influenced by the movies connected to that user.

After training, users and movies are represented as 32-dimensional vectors.

Movies with similar embeddings are considered good recommendation candidates.

## Technologies

* Python
* PyTorch
* PyTorch Geometric
* GraphSAGE
* FAISS
* NumPy
* Pandas
* scikit-learn
* MovieLens-1M

## Project Structure


graph-recommender/
│
├── data/
│   ├── ml-1m/
│   │   ├── ratings.dat
│   │   ├── movies.dat
│   │   └── users.dat
│   │
│   ├── graph.pt
│   └── embeddings.pt
│
├── src/
│   ├── prepare_data.py
│   ├── model.py
│   ├── train.py
│   ├── recommend.py
│   ├── evaluate.py
│   └── mf_baseline.py
│
├── requirements.txt
└── README.md


## Graph Construction

The MovieLens interactions are converted into a graph.

User nodes are assigned internal node IDs first, followed by movie nodes.

Example:

0       → User 1
1       → User 2
...
6040    → Movie 1
6041    → Movie 2
...

An interaction creates edges in both directions:

User → Movie
Movie → User


This allows GraphSAGE to propagate information between users and movies.

## GraphSAGE Model

The model contains:

```python
self.embedding = torch.nn.Embedding(
    num_nodes,
    hidden_dim
)

self.conv1 = SAGEConv(
    hidden_dim,
    hidden_dim
)

self.conv2 = SAGEConv(
    hidden_dim,
    embedding_dim
)
```

The data flow is:

Node IDs
   |
   v
64-dimensional initial embeddings
   |
   v
GraphSAGE Layer 1
   |
   v
64-dimensional representations
   |
   v
ReLU
   |
   v
GraphSAGE Layer 2
   |
   v
32-dimensional final embeddings


Each GraphSAGE layer uses `edge_index` to determine which nodes are neighbors.

Conceptually:


Node representation
        +
Neighbor representations
        |
        v
   Aggregation
        |
        v
New node representation


With two GraphSAGE layers, information can propagate across approximately two hops in the graph.

## Training

The model is trained using positive and negative user-movie pairs.

A positive pair represents an observed interaction:


User 10 → Movie 100


A negative pair is created by sampling a movie that the user did not interact with.

The model calculates:

positive score = user_embedding · movie_embedding

negative score = user_embedding · negative_movie_embedding


The training objective encourages:

positive score > negative score

The embeddings are then saved to:


data/embeddings.pt

## Recommendation with FAISS

After training, movie embeddings are added to a FAISS index.

The embeddings are L2-normalized:

```python
faiss.normalize_L2(movie_embeddings)
```

Then an inner-product index is created:

```python
index = faiss.IndexFlatIP(dimension)
```

Because the vectors are normalized, inner product is equivalent to cosine similarity.

For a user:

User embedding
      |
      v
     FAISS
      |
      v
Most similar movie embeddings
      |
      v
Top-K recommendations


## Evaluation

The recommender is evaluated using:

### Recall@20

Measures whether the relevant test movie appears anywhere in the top 20 recommendations.

Correct movie in Top 20 → 1
Correct movie not in Top 20 → 0


The final Recall@20 is averaged across evaluated users.

### NDCG@20

Measures both:

1. Whether the correct movie was recommended.
2. How highly it was ranked.

A correct movie at rank #1 receives a higher score than the same movie at rank #20.


Higher NDCG@20
        |
        v
Better ranking of relevant items


## Matrix Factorization Baseline

A matrix-factorization-style baseline is included to provide a comparison with the graph-based approach.

The comparison is:


Matrix Factorization
        vs
GraphSAGE


using:

Recall@20
NDCG@20

Example results table:

| Model                | Recall@20 | NDCG@20 |
| -------------------- | --------: | ------: |
| Matrix Factorization |    X.XXXX |  X.XXXX |
| GraphSAGE            |    X.XXXX |  X.XXXX |

## Installation

Install the required packages:

```bash
pip install torch
pip install torch-geometric
pip install faiss-cpu
pip install pandas numpy scipy scikit-learn
```

> On some Python/OS combinations, FAISS installation may require a compatible Python version or a Conda installation.

## Running the Project

### 1. Prepare the graph

```bash
python src/prepare_data.py
```

This creates:

data/graph.pt

### 2. Train GraphSAGE

```bash
python src/train.py
```

This creates:

data/embeddings.pt

### 3. Generate recommendations

```bash
python src/recommend.py
```

### 4. Evaluate GraphSAGE

```bash
python src/evaluate.py
```

### 5. Run the baseline

```bash
python src/mf_baseline.py
```

## Example Recommendation


Recommendations for User 1
----------------------------------------
1. Star Wars: Episode V - The Empire Strikes Back
2. Star Wars: Episode IV - A New Hope
3. Indiana Jones and the Last Crusade
...


## Key Concepts Demonstrated

This project demonstrates:

* Graph representation of user-item interactions
* Node embeddings
* Graph Neural Networks
* GraphSAGE neighbor aggregation
* Negative sampling
* Embedding-based recommendation
* Approximate nearest-neighbor retrieval
* FAISS
* Cosine similarity
* Recall@K
* NDCG@K
* Matrix-factorization baseline

## Future Improvements

Potential improvements include:

* Proper train/validation/test graph splitting
* Better negative sampling
* Mini-batch GraphSAGE training
* PinSage-style importance sampling
* Approximate FAISS indexes such as IVF/HNSW
* Larger datasets
* Hyperparameter tuning
* Cold-start handling
* Online recommendation serving API

## Disclaimer

This project is a lightweight **PinSage-inspired GraphSAGE recommender**, designed to demonstrate the core ideas of graph-based recommendation rather than reproduce Pinterest's production PinSage system.
