import threading
import time
import math

class Timer(threading.Timer):

    def start(self):
        self.started_at = time.time()
        threading.Timer.start(self)
    def remaining(self):
        return math.floor(self.interval - (time.time() - self.started_at))
    def elapsed(self):
        return time.time() - self.started_at