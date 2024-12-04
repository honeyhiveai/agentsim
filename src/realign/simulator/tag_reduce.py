import asyncio
import numpy as np
import os
import json
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from openai import AsyncOpenAI
import matplotlib.pyplot as plt
import umap

aclient = AsyncOpenAI()

# Use a more suitable embedding model
async def embed_tags(tags):
    tasks = [aclient.embeddings.create(
        model='text-embedding-ada-002',  # Better for phrases and sentences
        input=str(tag),
    ) for tag in tags]
    
    responses = await asyncio.gather(*tasks)
    embeddings = np.array([response.data[0].embedding for response in responses])
    return embeddings

async def main():

    # Step 1: Compute Embeddings
    # tags_data = json.load(open(os.path.join(os.path.dirname(__file__), 'tags.json')))
    # tags = [tag for x in tags_data['tags'] for tag in x['persona_tags']]  # List of 100 tags
    # tags = tags_data['tags']
    import pandas as pd
    
    # Read tags from CSV file
    tags_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'tags.csv'))
    tags = tags_df['tags'].tolist()

    embeddings = await embed_tags(tags)  # Shape: (100, embedding_dim)

    # Step 2: Normalize Embeddings
    embeddings_normalized = normalize(embeddings)

    # Step 3: Clustering with Agglomerative Clustering
    n_clusters = 10
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(embeddings_normalized)

    # Step 4: Select Representative Tags
    representative_tags = []
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_embeddings = embeddings_normalized[cluster_indices]
        
        # Compute pairwise cosine distances within the cluster
        pairwise_dist = cosine_distances(cluster_embeddings)
        # Sum distances for each tag
        total_distances = pairwise_dist.sum(axis=1)
        # Select the tag with the minimal total distance to others
        closest_index = cluster_indices[np.argmin(total_distances)]
        representative_tags.append(tags[closest_index])

    # Step 5: Result
    print("Representative Tags:", representative_tags)
    clusters = {}

    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_tags = [tags[index] for index in cluster_indices]
        clusters[i] = cluster_tags

    # Print each cluster and its tags
    for cluster_id, cluster_tags in clusters.items():
        print(f"Cluster {cluster_id} / {representative_tags[cluster_id]}:")
        for tag in cluster_tags:
            print(f" - {tag}")
        print("\n")

    # Reduce dimensions for visualization
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform(embeddings_normalized)

    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=cluster_labels, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(n_clusters)))
    plt.title('Tag Embeddings Clustering Visualization')
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())