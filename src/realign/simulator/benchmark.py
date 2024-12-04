import time

import matplotlib.pyplot as plt
import numpy as np
from openai import AsyncOpenAI
import tiktoken
import os
import pandas as pd
import asyncio


client = AsyncOpenAI()
lock = asyncio.Lock()

def generate_input_text(num_tokens, model):
    
    instruction = "Summarize the text then write an essay:\n" # exactly 10 tokens
    num_tokens = num_tokens - 10
    
    with open(os.path.join(os.path.dirname(__file__), 'art_of_war.txt'), 'r') as file:
        text = file.read()
        text = text + text + text + text # 400k tokens
    
    shift = np.random.randint(0, len(text))
    text = text[shift:] + text[:shift]
    tokenizer = tiktoken.encoding_for_model(model)
    token_ids = tokenizer.encode(text)[:num_tokens]
    return instruction + tokenizer.decode(token_ids)


async def call_openai(model, input_text, output_tokens):
    start_time = time.perf_counter()
        
    await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": input_text}],
        max_tokens=output_tokens
    )
    
    end_time = time.perf_counter()
    return end_time - start_time

async def run_single_benchmark(
    runs=100,
    model="gpt-4o-mini",
    input_tokens=8000,
    output_tokens=1000
):
    
    print(f"\nBenchmark Params: Model {model}; Input Tokens {input_tokens}; Output Tokens {output_tokens}; Number of Runs {runs}")
    
    input_text = generate_input_text(input_tokens, model)
    
    tasks = [
        call_openai(model, input_text, output_tokens)
        for _ in range(runs)
    ]
    times = await asyncio.gather(*tasks)
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    p50 = np.percentile(times, 50)
    p90 = np.percentile(times, 90)
    p95 = np.percentile(times, 95)
    
    # print("\nBenchmark Results:")
    # print(f"Average Time: {avg_time:.4f} seconds")
    # print(f"Minimum Time: {min_time:.4f} seconds")
    # print(f"Maximum Time: {max_time:.4f} seconds")
    # print(f"Standard Deviation: {std_time:.4f} seconds")
    # print(f"50th Percentile: {p50:.4f} seconds")
    # print(f"90th Percentile: {p90:.4f} seconds")
    # print(f"95th Percentile: {p95:.4f} seconds")
    
    
    results = {
        "model": [model],
        "input_tokens": [input_tokens],
        "output_tokens": [output_tokens],
        "num_calls": [runs],
        "avg_time": [avg_time],
        "min_time": [min_time],
        "max_time": [max_time],
        "std_time": [std_time],
        "p50": [p50],
        "p90": [p90],
        "p95": [p95]
    }
    
    async with lock:
        current_path = os.path.dirname(os.path.abspath(__file__))
        df = pd.DataFrame(results)
        csv_path = os.path.join(current_path, 'benchmark_results.csv')
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        df = df.sort_values(by='p95', ascending=True)
        df.to_csv(csv_path, index=False)
    
    # Plotting the distribution with more granular bins
    # plt.hist(times, bins=50, edgecolor='black')  # Increased number of bins for granularity
    # plt.title('Distribution of API Call Times')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Frequency')
    # plt.show()


async def run_benchmarks(
    runs=[100],
    models=["gpt-4o-mini", "gpt-4o"],
    out_token_range=[100, 200, 400, 800, 1600],
):
    tasks = []
    for run in runs:
        for model in models:
            for output_tokens in out_token_range:
                input_tokens = 8 * output_tokens # [800, 1600, 3200, 6400, 12800]
                tasks.append(
                    run_single_benchmark(run, model, input_tokens, output_tokens)
                )
    # await asyncio.gather(*tasks)
    for task in tasks:
        await task

if __name__ == "__main__":
    # asyncio.run(run_benchmarks())
    
    # open the csv into a df and show me the p95 by input tokens and model. Plot a bar chart.
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'benchmark_results.csv'))
    df = df.sort_values(by='p95', ascending=True)
    plt.figure(figsize=(10, 6))
    
    # Get unique values for input tokens and models
    input_tokens = sorted(df['input_tokens'].unique())    
    # Set width of bars and positions of the bars
    bar_width = 0.35
    x = np.arange(len(input_tokens))
    
    # Plot bars for each model
    # Plot each data point as a separate bar
    for i, row in df.iterrows():
        model_idx = list(df['model'].unique()).index(row['model'])
        input_token_idx = list(input_tokens).index(row['input_tokens'])
        plt.bar(input_token_idx + model_idx*bar_width, row['p95'], bar_width, label=row['model'])
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Customize the plot
    plt.xlabel('Input Tokens')
    plt.ylabel('P95 Latency (seconds)')
    plt.title('P95 Latency by Model and Input Size')
    plt.xticks(x + bar_width/2, input_tokens)
    plt.legend()
    
    plt.tight_layout()
    plt.show()