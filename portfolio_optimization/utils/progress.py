import mmap
import struct
import os
import time


class ProgressLogger:
    def __init__(self, shared_file_path="./portfolio_progress"):
        with open("./portfolio_progress", "wb") as f:
            f.write(b"\x00" * 9)  # writing 9 bytes of zeros to the file
        self.shared_file_path = shared_file_path

    # Remove the shared file
    def delete(self):
        os.remove(self.shared_file_path)

    def context(self, context_name: str):
        return TaskContext(context_name, self.shared_file_path, self)

    def print_progress(self):
        with open(self.shared_file_path, "r+b") as f, mmap.mmap(
            f.fileno(), 9
        ) as shared_state:
            while struct.unpack("b", shared_state[8:9])[
                0
            ]:  # wait until no other process is reading/writing
                time.sleep(0.01)  # sleep for 10 milliseconds
            shared_state[8:9] = struct.pack("b", 1)  # mark as reading
            progress = (
                100
                * struct.unpack("i", shared_state[4:8])[0]
                / struct.unpack("i", shared_state[0:4])[0]
                if struct.unpack("i", shared_state[0:4])[0] > 0
                else 0
            )
            shared_state[8:9] = struct.pack("b", 0)  # unmark as reading
            print(f"[PROGRESS] {progress:.2f}%")


class TaskContext:
    def __init__(self, name: str, file_path: str, progress_logger: ProgressLogger):
        self.name = name
        self.file_path = file_path
        self.progress_logger = progress_logger

    def add_task(self, name: str, difficulty: int):
        task_name = f"{self.name}_{name}"
        with open(self.file_path, "r+b") as f, mmap.mmap(f.fileno(), 9) as shared_state:
            while struct.unpack("b", shared_state[8:9])[
                0
            ]:  # wait until no other process is reading/writing
                time.sleep(0.01)  # sleep for 10 milliseconds
            shared_state[8:9] = struct.pack("b", 1)  # mark as writing
            total_difficulty = struct.unpack("i", shared_state[0:4])[0] + difficulty
            shared_state[0:4] = struct.pack("i", total_difficulty)
            shared_state[8:9] = struct.pack("b", 0)  # unmark as writing

    def end_task(self, name: str):
        task_name = f"{self.name}_{name}"
        with open(self.file_path, "r+b") as f, mmap.mmap(f.fileno(), 9) as shared_state:
            while struct.unpack("b", shared_state[8:9])[
                0
            ]:  # wait until no other process is reading/writing
                time.sleep(0.01)  # sleep for 10 milliseconds
            shared_state[8:9] = struct.pack("b", 1)  # mark as writing
            finished_difficulty = struct.unpack("i", shared_state[4:8])[0] + 1
            shared_state[4:8] = struct.pack("i", finished_difficulty)
            shared_state[8:9] = struct.pack("b", 0)  # unmark as writing
