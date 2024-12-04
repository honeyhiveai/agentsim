from openai import AsyncOpenAI
import asyncio
import time
import matplotlib.pyplot as plt
import numpy as np
client = AsyncOpenAI()

async def make_api_call(call_number):
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{
                "role": "system", 
                "content": "You are a helpful assistant. In a single sentence, creatively describe the meaning of life."
            }],
            max_tokens=512
        )
        print(f"Call {call_number} completed")
    
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        return 'error'


async def evaluate_batch(batch):
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{
                "role": "system", 
                "content": "You are an astute judge. Rate this description of the meaning of life on a scale of 1 to 5. Respond in JSON format with the following format: {'rating': 2}."
            }],
            response_format={"type": "json_object"}
        )
        print(f"Eval {call_number} completed")
    
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        return 'error'

# initiate 10 generations and add to queue with async locking
# consume everything in the queue with async locking
# for each batch make an evaluate call




async def run_benchmark(n_calls=100):
    print(f"Starting {n_calls} parallel API calls, 3 times...")
    
    combined_latencies = []
    for i in range(3):
        print(f"Run {i+1}/3")
        # Create list of coroutines to run in parallel
        tasks = [make_api_call(j) for j in range(n_calls)]
        
        # Run all API calls concurrently
        latencies = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None values and exceptions
        latencies = [lat for lat in latencies if isinstance(lat, (int, float))]
        
        combined_latencies.extend(latencies)
    
    return combined_latencies

def plot_histogram(latencies):
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=20, edgecolor='black')
    plt.title('OpenAI API Latency Distribution')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Frequency')
    
    # Add statistics
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    stats_text = f'Mean: {mean_latency:.2f}s\nMedian: {median_latency:.2f}s\n95th percentile: {p95_latency:.2f}s'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig('api_latency_histogram.png')
    plt.close()

if __name__ == "__main__":
    latencies = asyncio.run(run_benchmark())
    plot_histogram(latencies)
    
    # Print summary statistics
    print("\nLatency Statistics (seconds):")
    print(f"Mean: {np.mean(latencies):.2f}")
    print(f"Median: {np.median(latencies):.2f}")
    print(f"95th percentile: {np.percentile(latencies, 95):.2f}")
    print(f"Min: {np.min(latencies):.2f}")
    print(f"Max: {np.max(latencies):.2f}")

# N = 100
# Mean: 13.24s
# Median: 12.84s
# 95th percentile: 18.42s
# Min: 7.17s
# Max: 23.06s

# 1024 tokens = 50 TPS
# Mean: 13.52s
# Median: 12.40s
# 95th percentile: 20.34s
# Min: 7.55s
# Max: 61.58s


# 512 tokens
# Mean: 8.85
# Median: 8.53
# 95th percentile: 12.84
# Min: 4.64
# Max: 20.97