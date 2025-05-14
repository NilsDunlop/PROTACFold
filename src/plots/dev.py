import time
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import signal

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, main_path):
        self.main_path = main_path
        self.process = None
        self.last_run_time = 0
        self.start_main_process() # Initial run

    def on_modified(self, event):
        # In some OS/editor combinations, multiple events can be triggered for a single save.
        # This debounce helps to only react once.
        if time.time() - self.last_run_time < 1.0: # 1 second debounce
            return
        
        if event.src_path.endswith('.py'):
            print(f"\n[Reloading] Code change detected in {event.src_path}")
            self.last_run_time = time.time()
            self.restart_main_process()

    def start_main_process(self):
        if self.process and self.process.poll() is None:
            print("[Info] Main process already running. This shouldn't happen if terminated correctly.")
            return

        try:
            print(f"\n[Running] Starting {self.main_path}...")
            env = os.environ.copy()
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            # Use Popen for non-blocking execution and process management
            self.process = subprocess.Popen([sys.executable, '-B', self.main_path], env=env)
            self.last_run_time = time.time()
        except Exception as e:
            print(f"Error starting {self.main_path}: {e}")
            self.process = None

    def stop_main_process(self):
        if self.process and self.process.poll() is None: # Check if process is running
            print(f"[Stopping] Terminating existing {self.main_path} process (PID: {self.process.pid})...")
            try:
                # Try to terminate gracefully
                if sys.platform == "win32":
                    self.process.send_signal(signal.CTRL_C_EVENT) # More like Ctrl+C
                else:
                    self.process.terminate() # Sends SIGTERM

                # Wait for a short period
                try:
                    self.process.wait(timeout=5) # Wait 5 seconds
                except subprocess.TimeoutExpired:
                    print(f"[Stopping] Process {self.process.pid} did not terminate, killing...")
                    self.process.kill() # Force kill if not terminated
                    self.process.wait() # Ensure it's killed
                print(f"[Stopping] Process {self.process.pid} terminated.")
            except Exception as e:
                print(f"Error stopping process {self.process.pid if self.process else 'Unknown'}: {e}")
            finally:
                self.process = None
        elif self.process: # Process exists but is not running
            self.process = None


    def restart_main_process(self):
        self.stop_main_process()
        # Brief pause to ensure resources are freed, especially network ports if main.py uses them.
        time.sleep(0.5)
        self.start_main_process()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_script_path = os.path.join(current_dir, "main.py")

    if not os.path.exists(main_script_path):
        print(f"Error: Could not find main.py at {main_script_path}")
        sys.exit(1)

    print(f"Found main.py at: {main_script_path}")

    # Clear .pyc files (optional, but can help ensure fresh loads)
    for root, _, files in os.walk(current_dir):
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    # print(f"Removed cached file: {file_path}")
                except OSError as e:
                    print(f"Warning: Could not remove {file_path}: {e}")

    event_handler = CodeChangeHandler(main_script_path)
    observer = Observer()
    observer.schedule(event_handler, path=current_dir, recursive=True)
    
    print(f"Watching for changes in Python files within {current_dir}. Press Ctrl+C to exit.")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Exiting] Stop signal received. Shutting down watcher and main process...")
        event_handler.stop_main_process() # Ensure main process is stopped on exit
        observer.stop()
    observer.join()
    print("[Exiting] Watcher and main process shut down.")