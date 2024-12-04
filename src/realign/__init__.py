# load this first
from .evaluators import evaluator, aevaluator

# instantiate all decorated evaluators in evallib
# from . import evallib

# load the agents
from .llm_utils import allm_messages_call, llm_messages_call, run_async, router

# load config
from .configs import config, assume_config, DEFAULT_CONFIG_PATH

# set and load the default config file
config.path = DEFAULT_CONFIG_PATH # defaults.yaml

# try finding and loading the config file
presumed_config = assume_config()
if presumed_config:
    config.path = presumed_config

# load the simulation classes
from .simulation import Simulation, ChatSimulation, Context

from jinja2 import Template
def render_messages(messages: list[dict], **kwargs):
    return [{
        'role': message['role'],
        'content': Template(message['content']).render(**kwargs)
    } for message in messages]

__all__ = [
    'configs',
    'config',
    'router',
    'load_config',
    'evaluator',
    'aevaluator',
    'render_messages',
    'llm_messages_call',
    'allm_messages_call',
    'run_async',
    'Simulation',
    'ChatSimulation',
    'Context',
]