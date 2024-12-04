from datasets import load_dataset
import threading

import asyncio
from itertools import cycle


import json
from realign import config
from openai import AsyncOpenAI
from realign.utils import render


class bcolors:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    LIGHT_GRAY = '\033[37m'
    DARK_GRAY = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


aclient = AsyncOpenAI()

persona_hub = load_dataset("proj-persona/PersonaHub", "persona")["train"]
shuffled_personas = persona_hub.shuffle()
persona_cycle = cycle(shuffled_personas)
persona_cycle_lock = threading.Lock()

def user_str(input_str):
    return f"{bcolors.BOLD}{bcolors.BLUE}You: {bcolors.RESET}{bcolors.BLUE}{input_str}"

def assistant_str(input_str):
    return f"{bcolors.BOLD}{bcolors.YELLOW}Assistant: {bcolors.RESET}{bcolors.YELLOW}{input_str}{bcolors.RESET}"

async def collect_feedback(conversation_history):

    while True:
        user_input = input(user_str("")); print(f"{bcolors.RESET}")    
        conversation_history.append({"role": "user", "content": user_input})
        
        if '/ok' in user_input.strip().lower():
            break
        
        # Create a chat completion with function calling
        response = await aclient.chat.completions.create(
            model=config.get_feedback.model,
            messages=conversation_history
        )
        assistant_message = response.choices[0].message.content

        print(assistant_str(assistant_message + f' {bcolors.DARK_GRAY}[say /ok to move on to input schema]{bcolors.RESET}'))
        conversation_history.append({"role": "assistant", "content": assistant_message})

async def bootstrap_inputs(conversation_history, input_shape):
    with persona_cycle_lock:
        persona = next(persona_cycle)['persona']
    
    response = await aclient.chat.completions.create(
        model=config.scenario.model,
        messages=[
            *conversation_history,
            {"role": "user", "content": render(config.scenario.scenario_prompt, persona=persona)}
        ],
    )
    scenario = response.choices[0].message.content
    
    response = await aclient.chat.completions.create(
        model=config.scenario.model,
        messages=[
            *conversation_history,
            {"role": "user", "content": render(config.scenario.input_factory_user, persona=persona, scenario=scenario, input_shape=input_shape)},
            {'role': 'assistant', 'content': render(config.scenario.input_factory_asst, persona=persona, scenario=scenario, input_shape=input_shape)},
            {'role': 'user', 'content': config.scenario.input_factory_json}
        ],
        response_format={"type": "json_object"}
    )
    input_json = json.loads(response.choices[0].message.content)
    
    return_dict = {
        'input': input_json,
        'metadata': {
            'scenario': scenario,
            'persona': persona
        }
    }
    
    # print(f"\n{bcolors.BOLD}{persona}{bcolors.RESET}")
    # print(f"{scenario}\n")
    # print(input_json, '\n')
    print('generated input')
    return return_dict
        
async def input_shape_agent(conversation_history):
    print(f"{bcolors.BOLD}{bcolors.BLUE}Generating input shape{bcolors.RESET}")
    final_output = {}
    while True:
        response = await aclient.chat.completions.create(
            model=config.scenario.model,
            messages=[
                *conversation_history,
                {"role": "user", "content": config.scenario.input_shape_writer}
            ],
        )
        input_shape = response.choices[0].message.content
        final_output = f"Please give feedback with '/ok' to confirm - {input_shape}"
        
        conversation_history.append({'role': 'assistant', 'content': final_output})
        print(f"{bcolors.BOLD}{bcolors.YELLOW}Assistant: {final_output}{bcolors.RESET}")
        
        # print sample outputs
        sample_outputs = '\n Sample Outputs:\n'
        inputs = await asyncio.gather(
            *[
                bootstrap_inputs(conversation_history.copy(), input_shape)
                for _ in range(3)
            ]
        )
        sample_outputs += '\n'.join([
            f"{i+1}. {json.dumps(input['input'], indent=4)}" 
            for i, input in enumerate(inputs)
        ])
        print(f"{bcolors.BOLD}{bcolors.GREEN}{sample_outputs}{bcolors.RESET}")
        
        user_input = input(user_str("")); print(f"{bcolors.RESET}")
        conversation_history.append({'role': 'user', 'content': user_input})
        if '/ok' in user_input.strip().lower():
            return final_output
        
    return final_output

async def generate_inputs(conversation_history, input_shape):
    num_inputs = int(input(f"\nHow many inputs would you like to generate? "))
    print(f'\nGenerating {num_inputs} inputs:\n')
    
    return await asyncio.gather(
        *[
            bootstrap_inputs(conversation_history, input_shape) 
            for _ in range(num_inputs)
        ]
    )
    
async def main():

    conversation_history = [
        {"role": "system", "content": config.get_feedback.system_prompt}
    ]
    print(f"{bcolors.BOLD}{bcolors.BLUE}Hello, I'm Beekeeper. I'm going to help you generate a synthetic dataset for your application. Tell me, what kind of software would you like to build?\n\n{bcolors.RESET}")
    
    await collect_feedback(conversation_history)
    # conversation

    input_shape = await input_shape_agent(conversation_history.copy())

    inputs = await generate_inputs(conversation_history.copy(), input_shape)

    file = {'inputs': inputs, 'input_shape': input_shape, 'conversation_history': conversation_history}

    # save to json file
    with open('inputs.json', 'w') as f:
        json.dump(file, f, indent=4)

# zero shot a dataset
# one shot a dataset

# slack bot
# CLI to start a process. github action?
# just run the backend and frontend on a server. docker container
# nextjs multi-window for chat view.


if __name__ == "__main__":
    asyncio.run(main())