from state import GlobalState
from event import Idea
from controller import Controller
from process import Process
from tsafepq import ThreadSafePriorityQueue

import random

from datasets import load_dataset
import threading
import curses
import time
import asyncio
from itertools import cycle
from typing import Optional
import contextlib
import io
import os
import json
from realign import config, render_messages

GlobalState.debug = False

persona_hub = load_dataset("proj-persona/PersonaHub", "math")["train"]
shuffled_personas = persona_hub.shuffle(seed=42)
persona_cycle = cycle(shuffled_personas)

from openai import AsyncOpenAI
client = AsyncOpenAI()

math_seed = next(persona_cycle)['synthesized text']

async def generate_sketch(persona, problem):
    response = await client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{
            "role": "system", 
            "content": f"Pretend that you are a clever mathematician. Based on the problem, generate a sketch of how you would solve this problem. Go over the main steps and keep your responses concise. Do NOT answer the question, just generate a rough sketch of the methology and steps. In the past, you have solved problems like this: {persona}. Use this as inspiration to guide your solution. Make sure you are being logical."
        },
        {
            "role": "user",
            "content": f"Produce the series of steps you would take to solve this problem: {problem}"
        }
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content


async def solve_problem(problem, sketch, i):
    messages = [
        {"role": "system", "content": "You are a mathematical problem solver. Your task is to solve a given problem using a chain of thought reasoning."},
        {"role": "user", "content": f"Solve this math problem: {problem}\nHere's a sketch of the solution you can use: {sketch}\nThink step by step and explain your reasoning. You should reason back and forth at least 3 times."}
    ]
    
    async def make_call(messages, json_mode=False):
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={'type': 'json_object'} if json_mode else None
        )
        return response

    while True:
        response = await make_call(messages)
        message = response.choices[0].message
        print(f'Run {i} got response. Chains of thought: {(len(messages) // 2) - 1}')
        # print('got response\n\n')
        
        if len(messages) > 8:
            messages.append({"role": "user", "content": "Time's up. Submit the final answer by responding with a JSON object with the key 'answer' and the value as your final answer with units, and with the key 'explanation' and the value as your explanation."})
            response = await make_call(messages, json_mode=True)
            resp_json = json.loads(response.choices[0].message.content)
            return resp_json["answer"], resp_json["explanation"], messages
        
        messages.append({"role": "assistant", "content": message.content})
        messages.append({"role": "user", "content": "Continue your reasoning. Make an observation about your current progress, and improve your reasoning."})
        # print('len(messages)', len(messages))
    
async def solve(problem, i):
    # math_seed = next(persona_cycle)['synthesized text']
    print(f'Run {i} math_seed created')
    
    sketch = await generate_sketch(math_seed, problem)
    print(f'Run {i} sketch created')
    
    answer, explanation, messages = await solve_problem(problem, sketch, i)
    
    print(f'Run {i} answer: {answer}')
    
    return math_seed, answer, explanation, messages

async def run_solves(problem, n=10):
    tasks = [solve(problem, i) for i in range(n)]
    results = await asyncio.gather(*tasks)
    seeds, answers, explanations, messages = zip(*results)

    # dump these into a json file (list of dicts)
    with open('answers.json', 'w') as f:
        json.dump([
            {'seed': seed, 'answer': answer, 'explanation': explanation, 'messages': messages} 
            for seed, answer, explanation, messages in zip(seeds, answers, explanations, messages)
        ], f)
    
    # print answers beautifully
    for i, answer in enumerate(answers, 1):
        print(f"{i}. {answer}")
        print()
        
    # make the pairs
    pairs = '\n'.join([f'Answer: {answer}\nExplanation: {explanation}' for answer, explanation in zip(answers, explanations)])
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {'role': 'user', 'content': f'Given the following answers and explanations for the problem: {problem}, rank the answers based on the quality and coherence of their explanations. Here are the answers and explanation pairs:\n\n{pairs}\n\nPlease format your response as follows: Ranked list of answers (from best to worst). You need not provide justification or additional text.'}
        ],
    )
    print('The ranked list of answers is:\n\n', response.choices[0].message.content)
        
    response = await client.chat.completions.create(
        model="o1-mini",
        messages=[
            {'role': 'user', 'content': f'Given the following answers and explanations for the problem: {problem}, summarize the answers and provide a final answer with uncertainty. You may discount any anomalous answers. Here are the answers and explanation pairs: {pairs}'}
        ],
    )
    
    return response.choices[0].message.content

problem = """
How many LLM tokens will be generated by all GPUs over the next 10 years? Think about the number of GPUs, the number of tokens generated per second per GPU, and the number of seconds in a year.
"""
problem = """
Estimate the total number of kilometers that all animals (excluding humans) have collectively walked, flown, or swum throughout Earth's history.
"""
final_answer = asyncio.run(run_solves(problem=problem, n=20))

print(final_answer)
