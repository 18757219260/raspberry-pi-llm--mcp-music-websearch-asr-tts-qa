import pyaudio
import webrtcvad
import time
from aip import AipSpeech

# 百度api
APP_ID = ''
API_KEY = '' 
SECRET_KEY = ''

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

class ASRhelper:
    def __init__(self):
        # 设置音频参数
        self.CHUNK = 480  
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.SILENCE_DURATION = 1.0  
        self.MAX_RECORD_SECONDS = 5  
        self.NO_SPEECH_TIMEOUT = 2.0  
        # self.voice = "zh-CN-XiaoyiNeural"

        self.vad = webrtcvad.Vad(2)  
     
        self.p, self.stream = self.get_audio_stream()

    def get_audio_stream(self):
        """初始化输入音频流"""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        return p, stream

    def real_time_recognition(self):
        """实时语音识别"""
        # print('*'*40,"可以说话咯😁","*"*40)

        #输入流
        input= []
        start_time = time.time()
        speech_started = False
        last_speech_time = time.time()

        while True:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                is_speech = self.vad.is_speech(data, self.RATE)

                if is_speech:
                    if  speech_started==False:
                        speech_started = True
                        # print('*'*10,"可以说话咯😁","*"*10)
                    last_speech_time = time.time()
                    input.append(data)
                else:
                    if speech_started:

                        if (time.time() - last_speech_time) >= self.SILENCE_DURATION:
                            print('*'*10,"语音结束🙊",'*'*10)
                            break
                if (time.time() - start_time) >= self.MAX_RECORD_SECONDS:
                    # print("录完了")
                    break

                if not speech_started and (time.time() - start_time) >= self.NO_SPEECH_TIMEOUT:
                    print("请你提出问题？😾")
                    start_time = time.time()  # Reset start time

            except Exception as e:
                print("录音有错误！！！", str(e))
                break

        if input:
            audio_data = b"".join(input)
            print(f"上传 {len(audio_data)} 个字节到🪰")
            result = client.asr(audio_data, 'pcm', self.RATE, {'dev_pid': 1537})
            if result['err_no'] != 0:
                # print("🧠 用户问：:", result['result'][0])

            
                print("❌ 识别失败:", result['err_msg'], "错误码:", result['err_no'])
        else:
            print("没有录到语音")
        
        return result

    def stop_recording(self):
        """关闭音频流"""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("音频流已关闭‼️")
    def main(self):
        try:
            self.real_time_recognition()
        except KeyboardInterrupt:
            print("*" * 10, "停止实时语音识别‼️","*" * 10)
        finally:
            assistant.stop_recording()

if __name__ == '__main__':
    assistant = ASRhelper()
    assistant.main()