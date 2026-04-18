from pynvml import *
from tensorflow.keras.callbacks import Callback

class GPUGuard(Callback):
    def __init__(self, max_usage_ratio=0.95):
        super().__init__()
        self.max_usage_ratio = max_usage_ratio
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
            self.active = True
        except Exception as e:
            print(f"GPU Guard inactive: {e}")
            self.active = False

    def on_epoch_end(self, epoch, logs=None):
        if self.active:
            info = nvmlDeviceGetMemoryInfo(self.handle)
            usage_ratio = info.used / info.total
            
            if usage_ratio > self.max_usage_ratio:
                print(f"\n[CRITICAL] GPU memory usage at {usage_ratio:.2%}. Stopping training to prevent OOM!")
                self.model.stop_training = True