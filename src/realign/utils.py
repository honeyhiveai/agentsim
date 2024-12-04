import asyncio
import warnings
import copy

from typing import Callable, Any, Coroutine, Optional
from jinja2 import Template
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
async def arun_callables(funcs: list[Callable], 
                         args: Optional[list[list]] = None, 
                         kwargs: Optional[list[dict]] = None) -> list[Any]:
    if args is None:
        args = [[]] * len(funcs)
    if kwargs is None:
        kwargs = [{}] * len(funcs)

    if len(funcs) != len(args) or len(funcs) != len(kwargs):
        raise ValueError("funcs, args, and kwargs must all be the same length.")

    async def run_task(index: int, 
                       func: Callable, 
                       arg: list, 
                       kwarg: dict) -> tuple[int, Any]:
        
        is_coro = (hasattr(func, '__call__') and asyncio.iscoroutinefunction(func.__call__)) or asyncio.iscoroutinefunction(func)
        
        if is_coro:
            result = await func(*arg, **kwarg)
        else:
            result = await asyncio.to_thread(func, *arg, **kwarg)
        return index, result

    tasks = [
        asyncio.create_task(run_task(i, func, arg, kwarg))
        for i, (func, arg, kwarg) in enumerate(zip(funcs, args, kwargs))
    ]

    results = await asyncio.gather(*tasks)
    return [result for _, result in sorted(results, key=lambda x: x[0])]

def render(text: str, **kwargs):
    return Template(text).render(**kwargs)

def run_async(
    tasks: list[Coroutine] | Coroutine,
    times: int | None = None
) -> Any | Coroutine:
    
    # run the tasks in parallel
    async def gather_async(*tasks):
        return await asyncio.gather(*tasks)
    
    async def single_task(task):
        if asyncio.iscoroutine(task):
            return await task
        else:
            return task
    
    async def run_multiple_times(coro, times):
        return await asyncio.gather(*[
            coro
            for _ in range(times)
        ])
    
    if not isinstance(tasks, (list, tuple)):
        coroutine = single_task(tasks)
    else:
        coroutine = gather_async(*tasks)
    
    if times is not None:
        coroutine = run_multiple_times(coroutine, times)
    
    # if we are in a running event loop, return coroutine to be awaited
    if asyncio.get_event_loop().is_running():
        return coroutine
    
    return asyncio.run(coroutine)

def try_import(module_name: str | None = None, 
               from_module: str | None = None, 
               import_names: str | list[str] | None = None) -> Any:
    """
    Attempts to import a module or specific objects from a module and returns them if successful.
    If the import fails, it raises a warning instead of an exception.

    Args:
        module_name (str): The name of the module to import. For example, 'realign.evallib.rag'
        from_module (str, optional): The module to import from, if using 'from x import y' syntax.
        import_names (str | list[str], optional): The specific object(s) to import, if using 'from x import y' syntax.

    Returns:
        Any: The imported module or object(s) if successful, None otherwise.
    """
    try:
        if from_module and import_names:
            if isinstance(import_names, str):
                import_names = [import_names]
            module = __import__(from_module, fromlist=import_names)
            if len(import_names) == 1:
                return getattr(module, import_names[0])
            else:
                return tuple(getattr(module, name) for name in import_names)
        else:
            return __import__(module_name)
    except ImportError:
        if from_module and import_names:
            names_str = ", ".join(import_names) if isinstance(import_names, list) else import_names
            warnings.warn(f"Failed to import {names_str} from {from_module}. Some functionality may be unavailable.", ImportWarning)
        else:
            warnings.warn(f"Failed to import {module_name}. Some functionality may be unavailable.", ImportWarning)
        return None

class dotdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        d = dict(*args, **kwargs)
        for k, v in d.items():
            self[k] = self._convert(v)

    def __getattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            return super().__getattr__(key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute or key '{key}'")

    def __setattr__(self, key, value):
        if key.startswith('__') and key.endswith('__'):
            super().__setattr__(key, value)
        else:
            self[key] = self._convert(value)

    def __setitem__(self, key, value):
        super().__setitem__(key, self._convert(value))

    def __delattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            super().__delattr__(key)
        else:
            del self[key]

    def __deepcopy__(self, memo):
        return dotdict(copy.deepcopy(dict(self), memo))

    def _convert(self, value):
        if isinstance(value, dict) and not isinstance(value, dotdict):
            return dotdict({k: self._convert(v) for k, v in value.items()})
        elif isinstance(value, list):
            return [self._convert(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(self._convert(item) for item in value)
        else:
            return value

    def __getattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            return super().__getattr__(key)
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, dotdict):
                value = dotdict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"key '{key}' not found in {self}")

    def __setattr__(self, key, value):
        if key.startswith('__') and key.endswith('__'):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, dotdict):
            value = dotdict(value)
        super().__setitem__(key, value)

    def __delattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            super().__delattr__(key)
        else:
            del self[key]

    def __deepcopy__(self, memo):
        return dotdict(copy.deepcopy(dict(self), memo))
