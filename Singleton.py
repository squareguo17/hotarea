import threading

class Singleton(object):
    _instance_lock = threading.Lock()

    def __init__(self, cls):
        self._cls = cls
        self.uniqueInstance = None

    def __call__(self):
        if self.uniqueInstance is None:
            with self._instance_lock:
                if self.uniqueInstance is None:
                    self.uniqueInstance = self._cls()
        return self.uniqueInstance