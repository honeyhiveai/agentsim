import threading
import json
import traceback

from event import Idea
import sys

class GlobalState:
    
    debug = False
    
    
    lock = threading.Lock()
    criteria = None
    queues: dict = {}
    processes: dict = {}
    messages = []
    stop_event = threading.Event()
        
    @staticmethod
    def update_criteria(new_criteria):
        with GlobalState.lock:
            GlobalState.criteria = new_criteria
            if GlobalState.debug:
                print(f"GlobalState: Updated criteria to {GlobalState.criteria}")
    
    @staticmethod
    def get_criteria():
        with GlobalState.lock:
            return GlobalState.criteria
    
    @staticmethod
    def add_queue(name: str, queue):
        with GlobalState.lock:
            GlobalState.queues[name] = queue
            if GlobalState.debug:
                print(f"GlobalState: Added queue '{name}'")
    
    @staticmethod
    def get_queue(name: str):
        with GlobalState.lock:
            return GlobalState.queues.get(name, None)
        
    @staticmethod
    def add_process(process):
        with GlobalState.lock:
            GlobalState.processes[process.name] = process
            if GlobalState.debug:
                print(f"GlobalState: Added process '{process.name}'")
            
    @staticmethod
    def get_process(name: str):
        with GlobalState.lock:
            return GlobalState.processes.get(name, None)

    @staticmethod
    def reset():
        with GlobalState.lock:
            GlobalState.criteria = None
            GlobalState.queues = {}
            GlobalState.processes = {}
            GlobalState.messages = []
            GlobalState.stop_event = threading.Event()
            if GlobalState.debug:
                print("GlobalState: Reset all state variables")
            
    @staticmethod
    async def rate_absolute(client, idea: "Idea", model: str = "gpt-4o-mini") -> float:
        prompt = f"Rate the following business idea on a scale of 1-5 for interestingness, viability, and uniqueness. Provide an overall rating that is the average of these three scores. Return your response in strict JSON format. For example, to return a rating of 3.5, return {{'overall_rating': 3.5}}. User criteria: {GlobalState.messages[-1]['content']}. Business idea: {idea.seed}"

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a creative business consultant specializing in personalized business ideas. Base your responses on the conversation between the user and assistant."},
                    {"role": "user", "content": prompt},
                    *GlobalState.messages,
                ],
                response_format={"type": "json_object"}
            )
            
            overall_rating = json.loads(response.choices[0].message.content.strip())['overall_rating']
            
            return float(overall_rating)
        
        except Exception as e:
            if GlobalState.debug:
                print(f"GlobalState: Error rating idea: {e}")
                print("Traceback:")
                traceback.print_exc()
            print(f"Error details: {type(e).__name__}: {str(e)}")
            return 0
    
    @staticmethod
    async def evolve_idea(client, idea: "Idea") -> "Idea":
        # evolve the idea based on its seed
        prompt_group = f"Given the group of personas '{idea.seed}', think about a problem they all might have in common, and suggest a unique and innovative business idea tailored to this group's characteristics and potential interests. Make sure you are basing the ideas based on my previous feedback. The business idea should be concise, creative, and aligned with the group's likely preferences and skills. IMPORTANT: Start with the common problem, and then build an elevator pitch of the business idea in one single sentence, and don't refer to the personas in your response."
        
        prompt = f"Given the persona '{idea.seed}', think about a problem they might have in common, and suggest a unique and innovative business idea tailored to this persona's characteristics and potential interests. Make sure you are basing the ideas based on my previous feedback. The business idea should be concise, creative, and aligned with the persona's likely preferences and skills. IMPORTANT: Start with the common problem, and then build an elevator pitch of the business idea in one single sentence, and don't refer to the persona in your response."

        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a creative and intelligent tech business founder from Columbia University with a variety of skills and experiences. Base your responses on the conversation between the user and assistant."},
                    {"role": "user", "content": prompt},
                    *GlobalState.messages,
                ],
                max_tokens=150
            )
            
            new_idea = response.choices[0].message.content.strip()
            
            evolved_idea = Idea(
                seed=new_idea,
                depth=idea.depth + 1,
                lineage=idea.lineage + [idea]
            )
            evolved_idea.abs_rating = await GlobalState.rate_absolute(client, evolved_idea)
            
            if GlobalState.debug:
                print(f"GlobalState: Evolved idea '{idea.seed}' into '{evolved_idea.seed}'")
            return evolved_idea
        
        except Exception as e:
            if GlobalState.debug:
                print(f"GlobalState: Error evolving idea: {e}")
            return idea 
    
    
    @staticmethod
    async def analyze_idea(client, idea: Idea) -> Idea:
        prompt = f"Analyze the following business idea in depth, putting yourself in the shoes of a Silicon Valley venture capitalist. Provide a short and concise 100-word analysis including potential market size, competition, unique selling points, and potential challenges. Business idea: {idea.seed}"
        
        try:
            response = await client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=1000
            )
            
            analysis = response.choices[0].message.content.strip()
            
            analyzed_idea = Idea(
                seed=analysis,
                depth=idea.depth + 1,
                lineage=idea.lineage + [idea],
            )
            analyzed_idea.abs_rating = await GlobalState.rate_absolute(client, analyzed_idea, model="gpt-4o")
            return analyzed_idea
        
        except Exception as e:
            print(f"GlobalState: Error analyzing idea: {e}", file=sys.stderr)
            return idea

