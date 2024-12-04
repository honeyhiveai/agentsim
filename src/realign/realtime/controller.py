
import time
from typing import Optional

from state import GlobalState

class Controller:
    """
    Manages the overall workflow by initializing queues, global state, and processes.
    """
    
    def start_processes(self, runtime_seconds: Optional[int] = None, process_names: Optional[list[str]] = None):
        """
        Starts all processes by initiating their threads.
        """
        if process_names:
            for name in process_names:
                if name not in GlobalState.processes.keys():
                    print(f"Controller: Process '{name}' not found. Skipping...")
                    continue
                
                process = GlobalState.get_process(name)
                process.start()
                print(f"Controller: Started process '{name}'.\n\n")
            return
        else:
            # start all processes
            for name, process in GlobalState.processes.items():
                
                print(f"\n\nController: Starting process '{name}'...")

                process.start()

                print(f"Controller: Started process '{name}'.\n\n")
            print(f"Controller: Started {len(GlobalState.processes)} processes.")
        
        # start an asyncio event loop
        # Start broadcasting in a separate thread
        # self.broadcast_thread = threading.Thread(target=asyncio.run, args=(self.broadcast_last_idea(sid, sio),))
        # self.broadcast_thread.start()
        
        # # Save the thread reference in the controller
        # self.broadcast_thread = self.broadcast_thread
        
        if runtime_seconds:
            time.sleep(runtime_seconds)
        else:
            while not GlobalState.stop_event.is_set():
                time.sleep(1)
    
    def stop_processes(self):
        """
        Signals all processes to stop and waits for them to finish.
        """
        GlobalState.stop_event.set()
        print(f"Controller: Stopping {len(GlobalState.processes)} processes...")
        time.sleep(1)
        for name, process in GlobalState.processes.items():
            # first check if the process is running
            if process.is_alive():
                process.join()
                print(f"Controller: Process '{name}' has been stopped.")
            
        print("Controller: All processes have been stopped.")
        
        # stop the broadcast thread
        # self.broadcast_thread.join()
        
        # After stopping, you can inspect the 'staging_queue'
        # staging_queue = GlobalState.get_queue('staging_queue')
        # if staging_queue:
        #     print("\nFinal Ideas in 'staging_queue':")
        #     while not staging_queue.is_empty():
        #         idea = staging_queue.poll()
        #         if idea:
        #             print(f" - {idea.seed} (Depth: {idea.depth})")
            

