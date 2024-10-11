import threading


class ThreadSafeArray:
    def __init__(self):
        self.array = []
        self.lock = threading.Lock()

    def append(self, value):
        with self.lock:
            self.array.append(value)
            print(f"Appended {value}")

    def remove(self, value):
        with self.lock:
            if value in self.array:
                self.array.remove(value)
                print(f"Removed {value}")
            else:
                print(f"Value {value} not found in array")

    def get(self, index):
        with self.lock:
            if index < len(self.array):
                return self.array[index]
            else:
                print(f"Index {index} out of range")
                return None

    def set(self, index, value):
        with self.lock:
            if index < len(self.array):
                self.array[index] = value
                print(f"Set index {index} to {value}")
            else:
                print(f"Index {index} out of range")

    def __len__(self):
        with self.lock:
            return len(self.array)

    def __str__(self):
        with self.lock:
            return str(self.array)
