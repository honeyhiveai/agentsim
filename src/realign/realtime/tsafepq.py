import threading
import heapq
import random
from typing import Any, Callable, Optional
from state import GlobalState

class ThreadSafePriorityQueue:
    
    def __init__(
        self, 
        heuristic_func: Optional[Callable[[Any], float]] = None,
        max_size: int = 20,
        stop_size: int = 100
    ):
        self._lock = threading.Lock()
        self._queue: list[tuple[float, Any]] = []
        self.heuristic_func = heuristic_func
        self.max_size = max_size
        self.stop_size = stop_size
        
        self.total_polls = 0
        
    def push(self, item: Any) -> None:
        """
        Pushes an item onto the queue, prioritized based on the heuristic function.
        
        Args:
            item (Any): The item to be added to the queue.
        """
        if self.heuristic_func:
            priority = self.heuristic_func(item)
        else:
            priority = 0
        
        with self._lock:
            heapq.heappush(self._queue, (priority, item))
            print(f"Pushed item: {item} with priority: {priority}")
            if len(self._queue) > self.max_size:
                heapq.heappop(self._queue)
            
    def poll(self) -> Optional[Any]:
        """
        Polls the highest-priority item from the queue.
        
        Returns:
            Optional[Any]: The highest-priority item, or None if the queue is empty.
        """
        with self._lock:
            if not self._queue:
                return None
            if self.total_polls > self.stop_size:
                GlobalState.stop_event.set()
            priority, item = heapq.heappop(self._queue)
            print(f"Polled item: {item} with priority: {priority}")
            self.total_polls += 1
            return item
        
    def poll_many(self, n: int) -> list[Any]:
        """
        Polls multiple items from the queue.
        """
        with self._lock:
            items = []
            for _ in range(n):
                if not self._queue:
                    break
                if self.total_polls > self.stop_size:
                    GlobalState.stop_event.set()
                    break
                priority, item = heapq.heappop(self._queue)
                print(f"Polled item: {item} with priority: {priority}")
                self.total_polls += 1
                items.append(item)
            return items

    def poll_random(self) -> Optional[Any]:
        """
        Polls a random item from the queue.
        
        Returns:
            Optional[Any]: A randomly selected item, or None if the queue is empty.
        """
        with self._lock:
            if not self._queue:
                return None
            index = random.randint(0, len(self._queue) - 1)
            priority, item = self._queue.pop(index)
            print(f"Randomly polled item: {item} with priority: {priority}")
            return item

    def peek(self) -> Optional[Any]:
        """
        Peeks at the highest-priority item without removing it.
        
        Returns:
            Optional[Any]: The highest-priority item, or None if the queue is empty.
        """
        with self._lock:
            if not self._queue:
                return None
            priority, item = self._queue[0]
            print(f"Peeked at item: {item} with priority: {priority}")
            return item
        
    def peek_many(self, n: int = 10) -> list[Any]:
        """
        Peeks at all items in the queue without removing them.
        """
        items = [item for _, item in reversed(self._queue)]
        return items[:n]

    def is_empty(self) -> bool:
        """
        Checks if the queue is empty.
        
        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        with self._lock:
            empty = len(self._queue) == 0
            return empty

    def size(self) -> int:
        """
        Returns the number of items in the queue.
        
        Returns:
            int: The size of the queue.
        """
        with self._lock:
            size = len(self._queue)
            return size

