from typing import Any
from .types import EvalResult, RunData
from .datasets import Dataset
from dotenv import load_dotenv
import asyncio
import json
from .base_class import BaseClass  # Import BaseClass

class Evaluation(BaseClass):  # Inherit from BaseClass
    
    async def subroutine(self) -> Any:
        raise NotImplementedError("Evaluation subroutine must be defined")

    async def subroutine_with_evals(self, run_id: int, **subroutine_kwargs) -> Any:
        
        if not self.subroutine:
            raise ValueError("Simulation subroutine must be defined")

        # run the simulation subroutine
        final_state = await self.subroutine(run_id, **subroutine_kwargs)

        # wrap the simulation run as an object
        sim_run_data = RunData(final_state, run_id=run_id)
        
        # save the run data
        self.run_data[run_id] = sim_run_data
        
        # run the evaluators
        eval_tasks = []
        for eval_func in self.evaluators:
            # pass object reference to the @evaluator decorator
            eval_tasks.append(asyncio.create_task(eval_func(sim_run_data)))

        # await all the evaluators
        evals: list[EvalResult] = await asyncio.gather(*eval_tasks)
        
        # save the evaluation results
        self.eval_results[run_id] = evals
    
    
    def __init__(self):
        super().__init__()
        self.dataset: Dataset = None

    def run(self) -> 'Evaluation':
        load_dotenv()

        if not self.dataset:
            raise ValueError("Evaluation dataset must be defined")

        # create an asyncio loop
        loop = asyncio.get_event_loop()

        tasks = [self.subroutine_with_evals(run_id=self.dataset.data['metadata'][i]['run_id']) for i in range(len(self.dataset.data['metadata']))]

        loop.run_until_complete(asyncio.gather(*tasks))

        return self
    
    # WIP: Clustering evaluations
    def cluster_evals(self) -> dict:
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            import pandas as pd
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

        except ImportError:
            print("Please install matplotlib and sklearn to visualize the clusters")
            return
 
        print(self.eval_results)
        
        embeddings = [evals[0].embedding.data[0]['embedding'] for _, evals in self.eval_results.items()]
        point_labels = [evals[0].explanation for _, evals in self.eval_results.items()]
        
        # Convert the list of embeddings to a 2D numpy arra
        df = pd.DataFrame({'embedding': embeddings, 'label': point_labels})
        df['embedding'] = df['embedding'].apply(lambda x: np.array(x))

        matrix = np.vstack(df.embedding.values)
        print(matrix.shape)
        
        n_clusters = 2

        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        kmeans.fit(matrix)
        labels = kmeans.labels_
        df["Cluster"] = labels

        # df.groupby("Cluster").Score.mean().sort_values()

        tsne = TSNE(n_components=2, perplexity=1, random_state=42, init="random", learning_rate=200)
        vis_dims2 = tsne.fit_transform(matrix)

        x = [x for x, y in vis_dims2]
        y = [y for x, y in vis_dims2]
        
        plt.figure(figsize=(20, 16))

        for category, color in enumerate(["purple", "green", "red", "blue"]):
            xs = np.array(x)[df.Cluster == category]
            ys = np.array(y)[df.Cluster == category]
            plt.scatter(xs, ys, color=color, alpha=0.3)

            avg_x = xs.mean()
            avg_y = ys.mean()

            plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

            # Add labels for each point
            # for i, (x, y) in enumerate(zip(xs, ys)):
            #     plt.annotate(df[df.Cluster == category]['label'].iloc[i], (x, y), xytext=(5, 5), 
            #                 textcoords='offset points', fontsize=8, alpha=0.7)  
          
        plt.title("Clusters identified visualized in language 2d using t-SNE")
        plt.show()

    # Removed export_eval_results method as it is implemented in BaseClass

    def visualization(self):
        print("Generating visualization...")
        # Placeholder for future implementation of more complex visualization
        return True

class ChatEvaluation(Evaluation):
    async def chat_evaluation_subroutine(self, run_id: int) -> Any:

        data = self.dataset.data

        # find the index in metadata where metadata['run_id'] == run_id
        datapoint = None
        for i in range(len(data['metadata'])):
            if data['metadata'][i]['run_id'] == run_id:
                datapoint = data['outputs'][i]
                break
        else:
            raise ValueError(f"Run ID {run_id} not found in the dataset")

        return datapoint['messages']
        
    
    def __init__(self, subroutine: Any = None):
        
        if not subroutine:
            self.subroutine = subroutine = self.chat_evaluation_subroutine

        super().__init__()
