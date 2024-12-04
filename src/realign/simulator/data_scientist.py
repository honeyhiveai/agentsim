import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import json
from realign import config
from openai import OpenAI, AsyncOpenAI

import asyncio

from realign.simulator.generate_inputs import bcolors, user_str, assistant_str
from realign.utils import render

client = OpenAI()
aclient = AsyncOpenAI()

async def enrich_data(df: pd.DataFrame, input_col: str, prompt: str, output_col: str, output_format: str):
    
    async def process_row(row):
        
        response = await aclient.chat.completions.create(
            model=config.data_scientist.enrich_data_model,
            messages=[
                {"role": "system", "content": config.data_scientist.enrich_data_prompt},
                {'role': 'user', 'content': render(config.data_scientist.enrichment_prompt, 
                                                   prompt=prompt, 
                                                   input=row[input_col], 
                                                   output_format=output_format)}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)['output']

    async def process_all():
        tasks = [process_row(row) for _, row in df[[input_col]].iterrows()]
        return await asyncio.gather(*tasks)

    # Run the coroutine using the event loop
    results = await process_all()
    
    df[output_col] = results
    
    return df

async def embed_data(df: pd.DataFrame, input_col: str, output_col: str):

    async def process_row_embedding(row, input_col):
        response = await aclient.embeddings.create(
            model='text-embedding-3-small',
            input=str(row[input_col]),
            dimensions=1024
        )
        return np.array(response.data[0].embedding)
    
    async def process_all():
        tasks = [process_row_embedding(row, input_col) for _, row in df[[input_col]].iterrows()]
        return await asyncio.gather(*tasks)

    # Run the coroutine using the event loop
    results = await process_all()
    
    df[output_col] = results
    
    return df

async def run_data_scientist(df, conversation_history):
    while True:
        user_input = input(user_str("")); print(f"{bcolors.RESET}")    
        conversation_history.append({"role": "user", "content": user_input})
        
        if '/run' in user_input.strip().lower():
            conversation_history.append({"role": "user", "content": user_input})
            
            # Try compiling the code to check for syntax errors before executing
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=config.data_scientist.model,
                        messages=[
                            *conversation_history,
                            {"role": "system", "content": render(config.data_scientist.generate_function, columns=df.columns)}
                        ],
                        response_format={"type": "json_object"}
                    )
                    func_json = json.loads(response.choices[0].message.content)

                    compile(func_json['code'], '<string>', 'exec')
                    break
                except SyntaxError as e:
                    print(f"{bcolors.RED}Syntax error in generated code (attempt {attempt + 1}/3): {e}{bcolors.RESET}")
                    if attempt == 2: 
                        print(f"{bcolors.RED}Failed after 3 attempts. Please try again.{bcolors.RESET}")
                        continue
            
            print(func_json['code'])
            
            namespace = {
                'pd': pd, 'np': np, 'plt': plt, 'TSNE': TSNE,
                'enrich_data': enrich_data, 'embed_data': embed_data,
            }
            
            try:
                exec(await func_json['code'], namespace)
                df = namespace['transform'](df)
                
                print("\nOutput:", df)
                conversation_history.append({"role": "assistant", "content": f"output successfully generated, df updated based on\n {func_json['code']}"})
            except Exception as e:
                print(f"{bcolors.RED}Error: {e}{bcolors.RESET}")
                import traceback
                print(f"{bcolors.RED}{traceback.format_exc()}{bcolors.RESET}")
                conversation_history.append({"role": "assistant", "content": f"error during transformation: {e}"})
        
        # Create a chat completion with function calling
        response = await aclient.chat.completions.create(
            model=config.data_scientist.model,
            messages=conversation_history
        )
        assistant_message = response.choices[0].message.content

        print(assistant_str(assistant_message + f' {bcolors.DARK_GRAY}[say /run to run transformation]{bcolors.RESET}'))
        conversation_history.append({"role": "assistant", "content": assistant_message})

async def main():

    with open('inputs.json', 'r') as f:
        data = json.load(f)

    # Extract inputs and normalize the nested dict into flattened columns
    df = pd.json_normalize(data['inputs'], sep='.')

    print(df.head())

    conversation_history = [
        {"role": "system", "content": render(config.data_scientist.collect_intent, columns=df.columns)}
    ]
    print(f"{bcolors.BOLD}{bcolors.BLUE}Hello, I'm Beekeeper. I'm going to help you analyze, transform, and visualize your dataset. What would you like to do? The following columns are available: " + "{',\n '.join(df.columns)}\n\n{bcolors.RESET}")

    await run_data_scientist(df, conversation_history)

if __name__ == "__main__":
    asyncio.run(main())