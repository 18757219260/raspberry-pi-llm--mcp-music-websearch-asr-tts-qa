import sys
import asyncio
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QTextEdit, QFrame, QScrollArea,
                           QShortcut, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QObject, QEvent
from PyQt5.QtGui import QFont, QPalette, QColor, QKeySequence, QIcon, QPixmap
import pyaudio
import webrtcvad
from aip import AipSpeech
import edge_tts
import io
import subprocess
import logging
import re
import json
from datetime import datetime
from collections import deque

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat.log"), logging.StreamHandler()]
)

# 百度ASR API配置
APP_ID = ''
API_KEY = '' 
SECRET_KEY = ''

# QA模型所需导入
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import nest_asyncio
from openai import OpenAI
import random
nest_asyncio.apply()

# ==================================
# 系统配置
# ==================================
# 树莓派模式配置
IS_COMPUTER_MODE = True  # 设置为False表示树莓派模式(实时说话)

# ==================================
# 对话管理器类 (从conversation.py)
# ==================================
class ConversationManager:
    def __init__(self, max_history=10, tracking_file="conversation_tracking.json"):
        # 对话历史管理
        self.conversation_history = deque(maxlen=max_history)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 性能跟踪指标
        self.tracking_data = {
            'session_id': self.session_id,
            'user_id': None,
            'start_time': time.time(),
            'total_questions': 0,
            'total_responses': 0,
            'avg_response_time': 0,
            'response_times': [],
            'error_count': 0,
            'conversation_log': []
        }
        
        self.tracking_file = tracking_file
        self._lock = asyncio.Lock()
    
    async def add_conversation_entry(self, question, answer, response_time=None):
        """异步添加对话记录"""
        async with self._lock:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer,
                'response_time': response_time
            }
            
            # 添加到历史记录
            self.conversation_history.append(entry)
            
            # 更新跟踪数据
            self.tracking_data['total_questions'] += 1
            self.tracking_data['total_responses'] += 1
            
            if response_time:
                self.tracking_data['response_times'].append(response_time)
                self.tracking_data['avg_response_time'] = sum(self.tracking_data['response_times']) / len(self.tracking_data['response_times'])
            
            # 轻量级日志记录（仅保存关键信息）
            log_entry = {
                'time': datetime.now().strftime("%H:%M:%S"),
                'q': question[:50] + '...' if len(question) > 50 else question,
                'response_time': round(response_time, 2) if response_time else None
            }
            self.tracking_data['conversation_log'].append(log_entry)
    
    async def record_error(self, error_type, error_message):
        """记录错误信息"""
        async with self._lock:
            self.tracking_data['error_count'] += 1
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': error_type,
                'message': str(error_message)[:100]
            }
            
            if 'errors' not in self.tracking_data:
                self.tracking_data['errors'] = []
            self.tracking_data['errors'].append(error_entry)
    
    def get_conversation_context(self, max_context=3):
        """获取最近的对话上下文"""
        recent_conversations = list(self.conversation_history)[-max_context:]
        context = []
        for conv in recent_conversations:
            context.append(f"问题: {conv['question']}")
            context.append(f"回答: {conv['answer']}")
        return "\n".join(context)
    
    async def save_tracking_data(self):
        """异步保存跟踪数据"""
        async with self._lock:
            try:
                with open(self.tracking_file, 'w', encoding='utf-8') as f:
                    json.dump(self.tracking_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.error(f"保存跟踪数据失败: {e}")
    
    def get_session_summary(self):
        """获取会话摘要"""
        duration = time.time() - self.tracking_data['start_time']
        return {
            'session_id': self.session_id,
            'duration': round(duration, 2),
            'total_questions': self.tracking_data['total_questions'],
            'avg_response_time': round(self.tracking_data['avg_response_time'], 2),
            'error_count': self.tracking_data['error_count']
        }

# ==================================
# 语音合成类 (从tts_stream.py)
# ==================================
class TTSStreamer:
    def __init__(self, voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.is_speaking = False
        self._lock = asyncio.Lock()
        self.mpg123_process = None
        self.speech_queue = asyncio.Queue()
        self.speech_task = None
        self._playback_complete = asyncio.Event()
        self._playback_complete.set()  # Initially set
        self._last_audio_time = 0

    def preprocess_text(self, text):
        """预处理文本，替换标点符号"""
        text = text.replace("，", ",")
        text = text.replace("。", ",")
        text = text.replace("、", ",")
        text = text.replace("；", ",")
        text = text.replace("：", ",")
        text = text.replace("*", ',')
        text = text.replace(".", ',')
        text = text.replace("#", ',')
        text = text.replace("？", ',')
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        return text

    async def start_player(self):
        """启动mpg123进程"""
        async with self._lock:
            if self.mpg123_process is None or self.mpg123_process.poll() is not None:
                try:
                    self.mpg123_process = subprocess.Popen(
                        ["mpg123", "-q", "-"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        bufsize=1024*8
                    )
                    logging.info("mpg123播放器已启动")
                except Exception as e:
                    logging.error(f"启动mpg123失败: {e}")
                    self.mpg123_process = None
                    raise

    async def stop_player(self):
        """安全关闭播放器进程"""
        async with self._lock:
            if self.mpg123_process:
                try:
                    self.mpg123_process.stdin.flush()
                    self.mpg123_process.stdin.close()
                    self.mpg123_process.terminate()
                    await asyncio.sleep(0.3)
                    if self.mpg123_process.poll() is None:
                        self.mpg123_process.kill()
                    await asyncio.sleep(0.2)
                    self.mpg123_process = None
                    logging.info("mpg123播放器已关闭")
                except Exception as e:
                    logging.error(f"关闭mpg123时出错: {e}")

    async def _generate_speech(self, text):
        """生成语音数据"""
        if not text or not text.strip():
            return None
            
        try:
            communicate = edge_tts.Communicate(
                text, 
                self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            audio_data = io.BytesIO()
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])
                    
            if audio_data.tell() > 0:
                audio_data.seek(0)
                return audio_data.getvalue()
            else:
                return None
        except Exception as e:
            logging.error(f"生成语音时出错: {e}")
            return None
            
    async def _speech_processor(self):
        """处理语音队列的后台任务，使用优化的延迟策略"""
        try:
            await self.start_player()
            
            while True:
                text = await self.speech_queue.get()
                
                if text is None:  # 结束信号
                    break
                    
                try:
                    self._playback_complete.clear()
                    self.is_speaking = True
                    
                    audio_data = await self._generate_speech(text)
                    
                    if audio_data:
                        if self.mpg123_process and self.mpg123_process.poll() is None:
                            self.mpg123_process.stdin.write(audio_data)
                            self.mpg123_process.stdin.flush()
                            
                            # 优化延迟策略 - 更准确的估算
                            text_length = len(text)
                            base_delay = 0.15  
                            min_delay = 0.8    
                            max_delay = 8.0    
                            
                            delay = max(min_delay, min(max_delay, text_length * base_delay))
                            await asyncio.sleep(delay + 0.3)  
                            
                        else:
                            await self.start_player()
                            if self.mpg123_process:
                                self.mpg123_process.stdin.write(audio_data)
                                self.mpg123_process.stdin.flush()
                                
                                text_length = len(text)
                                delay = max(0.8, min(8.0, text_length * 0.12))
                                await asyncio.sleep(delay + 0.3)
                    
                    self._last_audio_time = time.time()
                    
                except Exception as e:
                    logging.error(f"播放语音时出错: {e}")
                
                finally:
                    self.is_speaking = False
                    self._playback_complete.set()
                    self.speech_queue.task_done()
                
        except Exception as e:
            logging.error(f"语音处理任务出错: {e}")
        finally:
            await self.stop_player()
            
    async def start_speech_processor(self):
        """启动语音处理任务"""
        if self.speech_task is None or self.speech_task.done():
            self.speech_task = asyncio.create_task(self._speech_processor())
            
    async def stop_speech_processor(self):
        """停止语音处理任务"""
        if self.speech_task and not self.speech_task.done():
            await self.speech_queue.put(None)
            await self.speech_task
            self.speech_task = None

    async def speak_text(self, text, wait=False):
        """流式处理文本"""
        text = self.preprocess_text(text)
        
        # 简单分段，不要太复杂
        segments = []
        max_length = 50  # 每段最大长度
        
        # 按逗号分割
        parts = text.split(',')
        current_segment = ""
        
        for part in parts:
            if len(current_segment) + len(part) > max_length and current_segment:
                segments.append(current_segment.strip())
                current_segment = part
            else:
                current_segment += part + ","
                
        if current_segment:
            segments.append(current_segment.strip())
            
        # 如果没有分段，就作为整体
        if not segments:
            segments = [text]
            
        # 确保处理器运行
        await self.start_speech_processor()
        
        # 播放所有段落
        for segment in segments:
            if segment.strip():
                await self.speech_queue.put(segment)
                
        # 如果需要等待完成
        if wait:
            await self.wait_until_done()

    async def wait_until_done(self):
        """等待所有语音播放完成 - 使用更智能的策略"""
        # 等待队列清空
        if self.speech_queue.qsize() > 0:
            await self.speech_queue.join()
        
        # 等待最后一个音频播放完成
        await self._playback_complete.wait()
        
        # 减少额外等待时间，提高响应速度
        await asyncio.sleep(0.4)  # 从1.0减少到0.4秒

    async def shutdown(self):
        """清理资源"""
        await self.stop_speech_processor()
        await self.stop_player()

# ==================================
# 语音识别类 (从asr.py)
# ==================================
class ASRHelper:
    def __init__(self):
        # 设置音频参数
        self.CHUNK = 480  # 读取帧
        self.FORMAT = pyaudio.paInt16  # 符合百度api编码
        self.CHANNELS = 1  # 单声道
        self.RATE = 16000  # 采样率
        self.SILENCE_DURATION = 1.0  # 静音时长
        self.MAX_RECORD_SECONDS = 5  # 录音最长时间
        self.NO_SPEECH_TIMEOUT = 2.0  # 没有语音的超时时间

        self.vad = webrtcvad.Vad(3)  # 语言检测
        self.client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
        
        self.p = None
        self.stream = None
        self.is_recording = False
    
    def initialize_audio(self):
        """初始化音频流"""
        if self.p is None:
            self.p = pyaudio.PyAudio()
        
        if self.stream is None:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
    def close_audio(self):
        """关闭音频流"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self.p is not None:
            self.p.terminate()
            self.p = None
            logging.info("音频流已关闭")

    async def real_time_recognition(self, callback=None):
        """实时语音识别（树莓派模式）"""
        self.initialize_audio()
        self.is_recording = True
        
        audio_input = []
        start_time = time.time()
        speech_started = False
        last_speech_time = time.time()
        
        # 状态回调
        if callback:
            callback("listening")

        try:
            while self.is_recording:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                is_speech = self.vad.is_speech(data, self.RATE)

                if is_speech:
                    if not speech_started:
                        speech_started = True
                    last_speech_time = time.time()
                    audio_input.append(data)
                else:
                    if speech_started:
                        if (time.time() - last_speech_time) >= self.SILENCE_DURATION:
                            logging.info("语音结束")
                            break
                            
                if (time.time() - start_time) >= self.MAX_RECORD_SECONDS:
                    logging.info("达到最大录音时间")
                    break

                if not speech_started and (time.time() - start_time) >= self.NO_SPEECH_TIMEOUT:
                    if callback:
                        callback("waiting")
                    logging.info("没有检测到语音")
                    return None
                    
                # 让出控制权给主事件循环
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logging.error(f"录音出错: {e}")
            if callback:
                callback("error")
            return None
            
        finally:
            self.is_recording = False
            
        if callback:
            callback("processing")
            
        if audio_input:
            audio_data = b"".join(audio_input)
            logging.info(f"上传 {len(audio_data)} 个字节进行识别")
            
            result = await asyncio.to_thread(
                self.client.asr, audio_data, 'pcm', self.RATE, {'dev_pid': 1537}
            )
            
            if result['err_no'] == 0:
                recognized_text = result['result'][0]
                logging.info(f"识别结果: {recognized_text}")
                return recognized_text
            else:
                logging.error(f"识别失败: {result['err_msg']}, 错误码: {result['err_no']}")
                return None
        else:
            logging.info("没有录到语音")
            return None
    
    async def press_to_talk(self, callback=None):
        """按住说话模式（电脑版）"""
        self.initialize_audio()
        self.is_recording = True
        
        audio_input = []
        start_time = time.time()
        
        # 状态回调
        if callback:
            callback("listening")

        try:
            while self.is_recording:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_input.append(data)
                
                # 让出控制权给主事件循环
                await asyncio.sleep(0.01)
                
                if (time.time() - start_time) >= self.MAX_RECORD_SECONDS:
                    logging.info("达到最大录音时间")
                    break
                    
        except Exception as e:
            logging.error(f"录音出错: {e}")
            if callback:
                callback("error")
            return None
            
        finally:
            self.is_recording = False
            
        if len(audio_input) < 10:  # 太短的录音可能是误触
            logging.info("录音太短，忽略")
            return None
            
        if callback:
            callback("processing")
            
        if audio_input:
            audio_data = b"".join(audio_input)
            logging.info(f"上传 {len(audio_data)} 个字节进行识别")
            
            result = await asyncio.to_thread(
                self.client.asr, audio_data, 'pcm', self.RATE, {'dev_pid': 1537}
            )
            
            if result['err_no'] == 0:
                recognized_text = result['result'][0]
                logging.info(f"识别结果: {recognized_text}")
                return recognized_text
            else:
                logging.error(f"识别失败: {result['err_msg']}, 错误码: {result['err_no']}")
                return None
        else:
            logging.info("没有录到语音")
            return None
    
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False

# ==================================
# 知识问答类 (从qa_model_easy.py)
# ==================================
class LlamaCppEmbeddings(Embeddings):
    """自定义嵌入类，使用 llama.cpp 加载 GGUF 模型生成嵌入"""
    def __init__(self, model_path):
        from llama_cpp import Llama
        self.model = Llama(model_path=model_path, embedding=True)
    def embed_documents(self, texts):
        return [self.model.embed(text) for text in texts]
    def embed_query(self, text: str):
        return self.model.embed(text)

class KnowledgeQA:
    def __init__(
        self,
        faiss_index_path="/home/wuye/vscode/raspberrypi_5/faiss_index",
        temperature=0.3,
        k_documents=3,
        embedding_model_path="/home/wuye/vscode/raspberrypi_5/text2vec_base_chinese_q8.gguf",
        conversation_manager=None
    ):
        self.faiss_index_path = faiss_index_path
        self.k_documents = k_documents
        self.temperature = temperature
        self.embedding_model = LlamaCppEmbeddings(model_path=embedding_model_path)
        self.vectorstore = self._load_vectorstore_with_retry()
        self.unknown_responses = [
            "我不知道",
            "这个问题我无法回答",
            "抱歉我不太会",
            "我还不了解这方面。",
            "对不起，我没有这方面的资料。",
            "我不知道这个答案，不过你可以去问吴家卓",
            "好像不太会？",
            "我里个豆阿，你问出这么难的问题我怎么会呢？"
        ]
        
        self.conversation_manager = conversation_manager or ConversationManager()
        
        # 初始化Qwen API客户端
        self.client = OpenAI(
            api_key="",
            base_url="",
        )
        
        # 系统消息设置
        self.sys_msg = {
            "role": "system",
            "content": "你是一个甘薯专家。请根据提供的参考内容回答问题，回答内容简洁。如果参考内容中没有相关信息，请回答'我不知道'。"
        }

    def _load_vectorstore_with_retry(self, max_retries=3):
        for i in range(max_retries):
            try:
                return FAISS.load_local(self.faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
            except Exception as e:
                logging.warning(f"第{i+1}次加载 FAISS 失败: {e}")
                time.sleep(1)
        raise RuntimeError("加载向量存储失败")
    
    async def ask_stream(self, question, context=True):
        start_time = time.time()
        
        context = ""
        if context:
            context = self.conversation_manager.get_conversation_context(max_context=3)
        
        docs = await asyncio.to_thread(
            self.vectorstore.as_retriever(search_kwargs={"k": self.k_documents}).invoke,
            question
        )

        if not docs:
            result = "未检索到相关内容。"
   
            response_time = time.time() - start_time
            await self.conversation_manager.add_conversation_entry(question, result, response_time)
            yield result
            return
        
        query = "你是一个甘薯专家，请你以说话的标准回答，请你根据参考内容回答，回答输出为一段，回答内容简洁，如果参考内容中没有相关信息，请回答'{}'。".format(random.choice(self.unknown_responses))
        
        # 构建包含上下文的提示
        doc_context = "\n\n".join([d.page_content for d in docs])
        
        # 如果有对话历史，将其加入提示
        if context:
            prompt = f"对话历史:\n{context}\n\n参考内容:\n{doc_context}\n\n当前问题:\n{question}\n\n要求:{query}\n\n"
        else:
            prompt = f"参考内容:\n{doc_context}\n\n问题:\n{question}\n\n要求:{query}\n\n"
        
        # 使用Qwen API进行流式调用
        messages = [
            self.sys_msg,
            {"role": "user", "content": prompt}
        ]
        
        try:
            stream = self.client.chat.completions.create(
                model="qwen2.5-omni-7b",  
                messages=messages,
                temperature=self.temperature,
                max_tokens=400,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    yield content
            
            response_time = time.time() - start_time
            await self.conversation_manager.add_conversation_entry(question, full_response, response_time)
            await self.conversation_manager.save_tracking_data()
                    
        except Exception as e:
            error_msg = f"API调用出错: {e}"
            logging.error(f"API调用失败: {e}")
            await self.conversation_manager.record_error("API_ERROR", str(e))
            yield error_msg

# ==================================
# UI 组件类
# ==================================
class MessageBubble(QWidget):
    """优化的聊天气泡组件"""
    def __init__(self, text, is_user=False, parent=None):
        super().__init__(parent)
        self.text = text
        self.is_user = is_user
        self._msg_label = None
        
        # 为7寸屏幕优化的布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 8, 5, 8)
        layout.setSpacing(10)
        
        # 用户头像
        avatar = QLabel()
        avatar.setFixedSize(36, 36)
        avatar.setStyleSheet(f"""
            background-color: {"#9EE846" if is_user else "#5B89DB"}; 
            border-radius: 18px;
            color: white;
            font-weight: bold;
            text-align: center;
        """)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setText("我" if is_user else "薯")
        
        # 消息文本
        self._msg_label = QLabel(text)
        self._msg_label.setFont(QFont("微软雅黑", 14))
        self._msg_label.setWordWrap(True)
        self._msg_label.setStyleSheet(f"""
            background-color: {"#A4E75A" if is_user else "#FFFFFF"}; 
            color: #303030;
            border-radius: 12px;
            padding: 12px;
            margin: 2px;
        """)
        
        # 根据消息来源调整布局
        if is_user:
            layout.addStretch()
            layout.addWidget(self._msg_label)
            layout.addWidget(avatar)
        else:
            layout.addWidget(avatar)
            layout.addWidget(self._msg_label)
            layout.addStretch()
        
        # 设置最小高度
        self.setMinimumHeight(50)
        # 设置最大宽度 (屏幕适配)
        self._msg_label.setMaximumWidth(300)
    
    @property
    def msg_label(self):
        return self._msg_label
    
    def update_text(self, text):
        if self._msg_label:
            self._msg_label.setText(text)

class ChatArea(QScrollArea):
    """优化的聊天区域"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置滚动区域属性
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # 给滚动条设置样式
        self.setStyleSheet("""
            QScrollBar:vertical {
                border: none;
                background: rgba(0, 0, 0, 0.1);
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 0, 0, 0.2);
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # 创建容器小部件
        self.container = QWidget()
        self.container.setStyleSheet("background-color: #EDEDED;")
        
        # 创建垂直布局
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(16)  # 增加间距使界面更清爽
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # 设置滚动区域的小部件
        self.setWidget(self.container)
    
    def add_message(self, text, is_user=False):
        """添加新消息"""
        if not text and not is_user:  # 允许机器人添加空消息（作为占位符）
           bubble = MessageBubble("", is_user, self)
           self.layout.addWidget(bubble)
           QTimer.singleShot(100, lambda: self.scrollToBottom())
           return bubble
        elif not text:  # 用户消息不能为空
           return None
           
       # 创建气泡消息
        bubble = MessageBubble(text, is_user, self)
        self.layout.addWidget(bubble)
        
        # 使用Timer确保滚动在渲染后执行
        QTimer.singleShot(100, lambda: self.scrollToBottom())
        return bubble
    
    def scrollToBottom(self):
        """滚动到底部"""
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
    
    def update_bubble_widths(self, width):
        """更新所有气泡的宽度"""
        max_width = min(300, int(width * 0.65))  # 为小屏幕优化
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            if item and item.widget():
                bubble = item.widget()
                if hasattr(bubble, 'msg_label'):
                    bubble.msg_label.setMaximumWidth(max_width)

class StatusIndicator(QWidget):
    """语音状态指示器"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(15, 0, 15, 0)
        
        # 状态图标
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(20, 20)
        self.icon_label.setStyleSheet("background-color: #5B89DB; border-radius: 10px;")
        
        # 状态文本
        self.text_label = QLabel("等待语音输入...")
        self.text_label.setFont(QFont("微软雅黑", 12))
        self.text_label.setStyleSheet("color: #606060;")
        
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.text_label)
        self.layout.addStretch()
        
        # 初始状态
        self.set_waiting()
        
    def set_waiting(self):
        """设置等待状态"""
        self.text_label.setText("等待语音输入...")
        self.icon_label.setStyleSheet("background-color: #5B89DB; border-radius: 10px;")
        
    def set_listening(self):
        """设置监听状态"""
        self.text_label.setText("正在聆听...")
        self.icon_label.setStyleSheet("background-color: #FF5252; border-radius: 10px;")
        
    def set_processing(self):
        """设置处理状态"""
        self.text_label.setText("正在思考...")
        self.icon_label.setStyleSheet("background-color: #FFC107; border-radius: 10px;")
        
    def set_answering(self):
        """设置回答状态"""
        self.text_label.setText("已回答，请继续提问...")
        self.icon_label.setStyleSheet("background-color: #4CAF50; border-radius: 10px;")

    # ==================================
    # 主应用类
    # ==================================
class SignalBridge(QObject):
    """信号桥接类，用于异步通信"""
    status_changed = pyqtSignal(str)
    add_user_message = pyqtSignal(str)
    start_bot_message = pyqtSignal()
    update_bot_message = pyqtSignal(str)
    request_real_time_listening = pyqtSignal()

class SweetPotatoGUI(QMainWindow):
    def __init__(self, user_name="吴大王"):
        super().__init__()
        self.user_name = user_name
        self.user_avatar_path = None  # 用户头像路径
        self.current_bot_bubble = None  # 当前的机器人消息气泡
        
        # 初始化组件
        self.chat_area = ChatArea()
        self.status_indicator = StatusIndicator() 
        self.voice_btn = None
        
        # 初始化信号桥接
        self.bridge = SignalBridge()
        self.bridge.status_changed.connect(self.update_status)
        self.bridge.add_user_message.connect(self.add_question)
        self.bridge.start_bot_message.connect(self.start_bot_message)
        self.bridge.update_bot_message.connect(self.update_bot_message)
        self.bridge.request_real_time_listening.connect(self.start_real_time_listening)
        
        # 初始化辅助类
        self.conversation_manager = ConversationManager(max_history=10)
        self.qa_model = KnowledgeQA(conversation_manager=self.conversation_manager)
        self.asr_helper = ASRHelper()
        self.tts_streamer = TTSStreamer()
        
        # 异步任务属性
        self.current_tasks = []
        self.current_answer = ""
        self.is_processing = False
        
        # 初始化UI
        self.init_ui()
        
        # 启动事件循环
        self.setup_asyncio_event_loop()
        
        # 树莓派模式下自动开始监听
        if not IS_COMPUTER_MODE:
            QTimer.singleShot(500, self.auto_start_listening)
    
    def init_ui(self):
        # 设置窗口属性
        self.setWindowTitle("甘薯知识问答系统")
        self.showFullScreen()
        
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 设置标题栏
        header = QWidget()
        header.setStyleSheet("background-color: #FF7E1F;")  # 甘薯橙色
        header.setFixedHeight(60)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        
        # 标题左侧图标
        icon_label = QLabel()
        icon_label.setFixedSize(40, 40)
        icon_label.setStyleSheet("""
            background-color: white;
            border-radius: 20px;
            color: #FF7E1F;
            font-weight: bold;
            font-size: 18px;
        """)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setText("薯")
        
        # 标题文本
        title_label = QLabel("甘薯知识助手")
        title_label.setFont(QFont("微软雅黑", 18, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        
        # 用户信息区域
        user_area = QWidget()
        user_layout = QHBoxLayout(user_area)
        user_layout.setSpacing(10)
        user_layout.setContentsMargins(0, 0, 0, 0)
        
        # 用户头像
        self.user_avatar_label = QLabel()
        self.user_avatar_label.setFixedSize(40, 40)
        self.user_avatar_label.setStyleSheet("""
            background-color: white;
            border-radius: 20px;
            border: 2px solid #FFFFFF;
        """)
        self.user_avatar_label.setAlignment(Qt.AlignCenter)
        self.user_avatar_label.setText("头像")
        self.user_avatar_label.mousePressEvent = self.choose_avatar
        
        # 用户名称标签
        user_label = QLabel(self.user_name)
        user_label.setFont(QFont("微软雅黑", 14))
        user_label.setStyleSheet("color: white;")
        user_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        user_layout.addWidget(self.user_avatar_label)
        user_layout.addWidget(user_label)
        
        header_layout.addWidget(icon_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(user_area)
        
        # 底部语音状态栏
        footer = QWidget()
        footer.setFixedHeight(70 if IS_COMPUTER_MODE else 0)  # 树莓派模式不显示按钮
        footer.setStyleSheet("background-color: #F8F8F8; border-top: 1px solid #DDD;")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(15, 5, 15, 5)
        
        if IS_COMPUTER_MODE:
            # 语音按钮（仅电脑模式）
            self.voice_btn = QPushButton("按住 说话")
            self.voice_btn.setFont(QFont("微软雅黑", 14))
            self.voice_btn.setFixedHeight(50)
            self.voice_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF7E1F;
                    color: white;
                    border-radius: 25px;
                    padding: 5px 20px;
                }
                QPushButton:pressed {
                    background-color: #D45E00;
                }
            """)
            
            # 设置按钮行为
            self.voice_btn.installEventFilter(self)
            
            footer_layout.addStretch()
            footer_layout.addWidget(self.voice_btn)
            footer_layout.addStretch()
        
        # 添加所有组件到主布局
        main_layout.addWidget(header)
        main_layout.addWidget(self.status_indicator)
        main_layout.addWidget(self.chat_area, 1)
        main_layout.addWidget(footer)
        
        # 设置ESC键退出
        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.close)
        
        # 添加欢迎消息
        welcome_msg = f"您好，{self.user_name}！我是甘薯知识助手，请通过语音向我提问关于甘薯的问题。"
        self.chat_area.add_message(welcome_msg)
        
    def choose_avatar(self, event):
        """选择用户头像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择头像",
            "",
            "图片文件 (*.jpg *.jpeg *.png)"
        )
        
        if file_path:
            self.user_avatar_path = file_path
            # 加载并设置头像
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    36, 36, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.user_avatar_label.setPixmap(scaled_pixmap)
                self.user_avatar_label.setText("")  # 清除原有文本
    
    def auto_start_listening(self):
        """自动开始实时聆听 (树莓派模式)"""
        self.status_indicator.set_listening()
        self.add_task(self.continuous_listening_task())
    
    async def continuous_listening_task(self):
        """连续实时聆听任务（树莓派模式）"""
        while not IS_COMPUTER_MODE:  # 只在树莓派模式下执行
            try:
                # 确保TTS完全结束后再开始聆听
                if self.tts_streamer.is_speaking:
                    await self.tts_streamer.wait_until_done()
                    await asyncio.sleep(0.5)  # 额外等待确保语音完全结束
                
                # 清理音频缓冲
                await self.clear_audio_buffer()
                
                # 语音识别
                text = await self.asr_helper.real_time_recognition(
                    callback=lambda status: self.bridge.status_changed.emit(status)
                )
                
                if text and not self.is_processing:  # 防止重复处理
                    # 设置处理状态
                    self.is_processing = True
                    
                    # 添加用户问题
                    self.bridge.add_user_message.emit(text)
                    
                    # 开始机器人消息
                    self.bridge.start_bot_message.emit()
                    
                    # 重置当前回答
                    self.current_answer = ""
                    
                    # 流式处理回答
                    async for chunk in self.qa_model.ask_stream(text):
                        self.current_answer += chunk
                        self.bridge.update_bot_message.emit(self.current_answer)
                    
                    # 语音合成
                    await self.tts_streamer.speak_text(self.current_answer, wait=True)
                    
                    self.is_processing = False
                
                # 短暂暂停，防止CPU占用过高
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logging.error(f"连续聆听过程中出错: {e}")
                self.is_processing = False
                await asyncio.sleep(1)  # 出错后暂停一会再继续
    
    async def clear_audio_buffer(self):
        """清理音频缓冲区"""
        try:
            if hasattr(self.asr_helper, 'stream') and self.asr_helper.stream:
                await asyncio.sleep(0.1)
                while self.asr_helper.stream.get_read_available() > 0:
                    self.asr_helper.stream.read(self.asr_helper.CHUNK, exception_on_overflow=False)
                logging.info("音频缓冲区已清理")
        except Exception as e:
            logging.warning(f"清理音频缓冲区时出错: {e}")
    
    def setup_asyncio_event_loop(self):
        """设置异步事件循环"""
        # 创建asyncio事件循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # 创建一个定时器来处理asyncio事件
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_asyncio_events)
        self.timer.start(10)  
        
    def _process_asyncio_events(self):
        """处理asyncio事件循环中的待处理事件"""
        # 仅处理所有当前待处理的回调，不会再次调用run_forever
        self.loop.call_soon(lambda: None)  
        self.loop.stop()
        self.loop.run_forever()
            
    def add_task(self, coro):
        """添加异步任务"""
        # 使用loop.create_task而不是asyncio.create_task
        task = self.loop.create_task(coro)
        self.current_tasks.append(task)
        task.add_done_callback(lambda t: self.current_tasks.remove(t) if t in self.current_tasks else None)
        return task
    
    def eventFilter(self, obj, event):
        """事件过滤器，处理按钮按下和释放事件"""
        if obj == self.voice_btn and IS_COMPUTER_MODE:
            # 按住说话模式（仅电脑模式）
            if event.type() == QEvent.MouseButtonPress:
                if not self.is_processing:
                    self.start_press_to_talk()
            elif event.type() == QEvent.MouseButtonRelease:
                self.stop_recording()
                            
        return super().eventFilter(obj, event)
    
    def update_status(self, status):
        """更新状态指示器"""
        if status == "waiting":
            self.status_indicator.set_waiting()
        elif status == "listening":
            self.status_indicator.set_listening()
        elif status == "processing":
            self.status_indicator.set_processing()
        elif status == "answering":
            self.status_indicator.set_answering()
    
    def add_question(self, text):
        """添加用户问题"""
        if text and self.chat_area:
            self.chat_area.add_message(text, is_user=True)
            self.status_indicator.set_processing()
    
    def start_bot_message(self):
        """开始机器人消息"""
        self.current_bot_bubble = self.chat_area.add_message("", is_user=False)
    
    def update_bot_message(self, text):
        """更新机器人消息"""
        if self.current_bot_bubble:
            self.current_bot_bubble.update_text(text)
            self.chat_area.scrollToBottom()
    
    def start_press_to_talk(self):
        """开始按住说话模式"""
        if self.is_processing:
            return
            
        self.status_indicator.set_listening()
        self.add_task(self.press_to_talk_task())
    
    def start_real_time_listening(self):
        """开始实时聆听模式"""
        if self.is_processing:
            return
            
        self.status_indicator.set_listening()
        self.add_task(self.real_time_listening_task())
    
    def stop_recording(self):
        """停止录音"""
        self.asr_helper.stop_recording()
    
    async def press_to_talk_task(self):
        """按住说话任务"""
        try:
            self.is_processing = True
            
            # 语音识别
            text = await self.asr_helper.press_to_talk(
                callback=lambda status: self.bridge.status_changed.emit(status)
            )
            
            if not text:
                self.bridge.status_changed.emit("waiting")
                self.is_processing = False
                return
                
            # 添加用户问题
            self.bridge.add_user_message.emit(text)
            
            # 开始机器人消息
            self.bridge.start_bot_message.emit()
            
            # 重置当前回答
            self.current_answer = ""
            
            # 流式处理回答
            async for chunk in self.qa_model.ask_stream(text):
                self.current_answer += chunk
                self.bridge.update_bot_message.emit(self.current_answer)
            
            # 语音合成
            await self.tts_streamer.speak_text(self.current_answer)
            
            self.bridge.status_changed.emit("answering")
            
        except Exception as e:
            logging.error(f"处理语音输入出错: {e}")
        finally:
            self.is_processing = False
    
    def resizeEvent(self, event):
        """窗口大小改变时重新调整消息气泡宽度"""
        super().resizeEvent(event)
        # 添加安全检查以防止崩溃
        if hasattr(self, 'chat_area') and self.chat_area:
            self.chat_area.update_bubble_widths(self.width())
            
    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        # 取消所有任务
        for task in self.current_tasks:
            task.cancel()
            
        # 关闭语音识别和合成资源
        self.asr_helper.close_audio()
        
        # 停止asyncio事件循环处理
        self.timer.stop()
        
        # 保存会话数据
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.conversation_manager.save_tracking_data())
        
        super().closeEvent(event)

    # 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置全局样式
    app.setStyle("Fusion")
    
    window = SweetPotatoGUI("吴家卓")
    window.show()
    
    sys.exit(app.exec_())