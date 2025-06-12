import os
import subprocess
from gpiozero import Button
from signal import pause
import sys
import sys
import os
print("Python executable:", sys.executable)
print("Python path:", sys.path)
print("ENV:", os.environ.get("VIRTUAL_ENV"))

BUTTON_PIN = 17
process = None

def toggle():
    global process
    if process is None or process.poll() is not None:
        print("启动 APP.py ...")
        env_path = "/home/joe/chatbox/chatbox/bin/python3"
        app_path = "/home/joe/chatbox/main_stream.py"
        process = subprocess.Popen(
            [env_path, app_path],
            cwd="/home/joe/chatbox"
        )
    else:
        print("终止 APP.py ...")
        process.terminate()

button = Button(BUTTON_PIN, pull_up=False)
button.when_pressed = toggle

print("等待按钮操作 ...")
pause()
