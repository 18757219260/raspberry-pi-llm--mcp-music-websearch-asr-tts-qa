import sys
import asyncio
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QTextEdit, QFrame, QScrollArea,
                           QPushButton, QFileDialog)
from PySide6.QtCore import Qt, QSize, QTimer, Signal, QObject, QEvent
from PySide6.QtGui import QFont, QPalette, QColor, QKeySequence, QIcon, QPixmap, QShortcut
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
        """预处理文本，保留更多原始标点结构"""
        # 只替换中文标点为对应的英文标点，不全部替换为逗号
        text = text.replace("，", ",")
        text = text.replace("。", ".")  # 保留句号的结构
        text = text.replace("、", ",")
        text = text.replace("；", ";")  # 保留分号
        text = text.replace("：", ":")  # 保留冒号
        text = text.replace("？", "?")  # 保留问号
        text = text.replace("！", "!")  # 保留感叹号
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
                            base_delay = 0.18  
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
        """流式处理文本，使用更智能的句子分割"""
        text = self.preprocess_text(text)
        
        # 智能分段 - 在自然断句点分割
        segments = []
        # 根据句子结束标点（句号、问号、感叹号、分号）或较长的逗号分句进行分段
        sentence_pattern = r'(?<=[.!?;])\s+|(?<=,)\s+(?=\S{5,})'
        parts = re.split(sentence_pattern, text)
        
        max_length = 60  # 增加最大长度，允许更完整的句子
        
        # 进一步处理过长的段落
        for part in parts:
            if len(part) <= max_length:
                segments.append(part)
            else:
                # 处理过长的段落，尝试在逗号处分割
                comma_parts = part.split(',')
                current_segment = ""
                
                for comma_part in comma_parts:
                    if len(current_segment) + len(comma_part) > max_length and current_segment:
                        segments.append(current_segment.strip())
                        current_segment = comma_part
                    else:
                        if current_segment:
                            current_segment += ", " + comma_part
                        else:
                            current_segment = comma_part
                
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
        faiss_index_path="faiss_index",
        temperature=0.3,
        k_documents=3,
        embedding_model_path="model/text2vec_base_chinese_q8.gguf",
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
            base_url="")
        
        # 系统消息设置
        self.sys_msg = {
            "role": "system",                                                           
            "content": "回答简洁"
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
        
        query = "你是一个甘薯专家，请你以说话的标准回答，请你根据参考内容回答，回答输出为一段，回答内容简洁，如果参考内容中没有任何相关信息，请回答'{}'。".format(random.choice(self.unknown_responses))
        
        # 构建包含上下文的提示
        doc_context = "\n\n".join([d.page_content for d in docs])
        
        # 如果有对话历史，将其加入提示
        if context:
            prompt = f"对话历史:\n{context}\n\n参考内容:\n{doc_context}\n\n当前问题:\n{question}\n\n"#要求:{query}\n\n"
        else:
            prompt = f"参考内容:\n{doc_context}\n\n问题:\n{question}\n\n"#要求:{query}\n\n"
        
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

        # 头像路径
        self.avatar_path = "guzz.png"
        self.robot_path = "sweetpotato.jpg"

        # 为7寸屏幕优化布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 12)  
        layout.setSpacing(15)

        avatar_size = 60
        avatar_label = QLabel()
        avatar_label.setFixedSize(avatar_size, avatar_size)
        avatar_label.setAlignment(Qt.AlignCenter)
        avatar_label.setStyleSheet(f"""
            border-radius: 5px;
            background-color: #DDDDDD;
            border: 3px solid {"#4CAF50" if is_user else "#FF9800"};
        """)

        # 加载头像图像
        avatar_path = self.avatar_path if is_user else self.robot_path
        pixmap = QPixmap(avatar_path)

        scaled = pixmap.scaled(
            avatar_size, avatar_size,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )
        avatar_label.setPixmap(scaled)

        self._msg_label = QLabel(text)
        self._msg_label.setFont(QFont("微软雅黑", 16))  # 增大字体
        self._msg_label.setWordWrap(True)
        self._msg_label.setMaximumWidth(780)  # 增加最大宽度
        self._msg_label.setStyleSheet(f"""
            background-color: {"#A4E75A" if is_user else "#FFFFFF"};
            color: #303030;
            border-radius: 20px;
            padding: 15px 20px;
            border: 2px solid {"#8BC34A" if is_user else "#E0E0E0"};
        """)

        # 按消息来源设置左右布局
        if is_user:
            layout.addStretch()
            layout.addWidget(self._msg_label)
            layout.addWidget(avatar_label)
        else:
            layout.addWidget(avatar_label)
            layout.addWidget(self._msg_label)
            layout.addStretch()

        self.setMinimumHeight(80)

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
        
        # 美化滚动条样式
        self.setStyleSheet("""
            QScrollArea {
                background-color: #F5F5F5;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: rgba(0, 0, 0, 0.05);
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 0, 0, 0.15);
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(0, 0, 0, 0.25);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # 创建容器小部件
        self.container = QWidget()
        self.container.setStyleSheet("background-color: #F5F5F5;")
        
        # 创建垂直布局
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(20)  # 增加间距使界面更清爽
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # 设置滚动区域的小部件
        self.setWidget(self.container)
    

    def smooth_scroll_to_bottom(self):
        """更平滑地滚动到底部"""
        scrollbar = self.verticalScrollBar()
        current = scrollbar.value()
        maximum = scrollbar.maximum()
        
        # 如果已经接近底部，直接跳到底部
        if maximum - current < 100:
            scrollbar.setValue(maximum)
            return
            
        # 否则使用动画效果
        steps = 5
        step_size = (maximum - current) / steps
        
        for i in range(steps):
            def scroll_step(idx=i):
                new_val = min(current + (idx + 1) * step_size, maximum)
                scrollbar.setValue(int(new_val))
            
            QTimer.singleShot(30 * (i + 1), scroll_step)
    
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
        max_width = min(800, int(width * 0.7))  
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
        self.setFixedHeight(50)
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border-bottom: 2px solid #E0E0E0;
            }
        """)
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(20, 10, 20, 10)
        
        # 状态图标
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(24, 24)
        self.icon_label.setStyleSheet("""
            background-color: #5B89DB; 
            border-radius: 12px;
            border: 2px solid white;
        """)
        
        # 状态文本
        self.text_label = QLabel("正在初始化...")
        self.text_label.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.text_label.setStyleSheet("color: #333333;")
        
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.text_label)
        self.layout.addStretch()
        
        # 初始状态
        self.set_waiting()
        
    def set_waiting(self):
        """设置等待状态"""
        self.text_label.setText("等待语音输入...")
        self.icon_label.setStyleSheet("""
            background-color: #5B89DB; 
            border-radius: 12px;
            border: 2px solid white;
        """)
        
    def set_listening(self):
        """设置监听状态"""
        self.text_label.setText("正在聆听...")
        self.icon_label.setStyleSheet("""
            background-color: #F44336; 
            border-radius: 12px;
            border: 2px solid white;
        """)
        
    def set_processing(self):
        """设置处理状态"""
        self.text_label.setText("正在思考...")
        self.icon_label.setStyleSheet("""
            background-color: #FFC107; 
            border-radius: 12px;
            border: 2px solid white;
        """)
        
    def set_answering(self):
        '''设置回答状态'''
        self.text_label.setText("正在播放欢迎语...")
        self.icon_label.setStyleSheet("""
            background-color: #E91E63; 
            border-radius: 12px;
            border: 2px solid white;
        """)
        
    def set_answerd(self):
        """设置回答状态"""
        self.text_label.setText("正在回答中...")
        self.icon_label.setStyleSheet("""
            background-color: #4CAF50; 
            border-radius: 12px;
            border: 2px solid white;
        """)

# ==================================
# 主应用类
# ==================================
class SignalBridge(QObject):
    """信号桥接类，用于异步通信"""
    status_changed = Signal(str)
    add_user_message = Signal(str)
    start_bot_message = Signal()
    update_bot_message = Signal(str)
    request_real_time_listening = Signal()


class SweetPotatoGUI(QMainWindow):
    def __init__(self, user_name="吴大王"):
        super().__init__()
        self.user_name = user_name
        self.current_bot_bubble = None
        
        self.follow_up_prompts = [
    "您还有什么问题吗？",
    "您还有什么想问的？",
    "您还想了解些什么？",
    "还有其他关于甘薯的问题吗？",
    "想成为吴家卓吗？",
    "还有什么疑问呢",
    "嘿嘿嘿你说呀？",
    "太豆了你，赶紧说？"
]
   
        



        # 初始化组件
        self.chat_area = ChatArea()
        self.status_indicator = StatusIndicator()

        # 信号桥接
        self.bridge = SignalBridge()
        self.bridge.status_changed.connect(self.update_status)
        self.bridge.add_user_message.connect(self.add_question)
        self.bridge.start_bot_message.connect(self.start_bot_message)
        self.bridge.update_bot_message.connect(self.update_bot_message)
        self.bridge.request_real_time_listening.connect(self.start_real_time_listening)

        # 辅助
        self.conversation_manager = ConversationManager(max_history=10)
        self.qa_model = KnowledgeQA(conversation_manager=self.conversation_manager)
        self.asr_helper = ASRHelper()
        self.tts_streamer = TTSStreamer()

        # 异步属性
        self.current_tasks = []
        self.current_answer = ""
        self.is_processing = False

        # UI 与事件循环
        self.init_ui()
        self.setup_asyncio_event_loop()
        # 播放欢迎并开始流程
        self.add_task(self.play_welcome_and_listen())

    def init_ui(self):
        self.setWindowTitle("甘薯知识问答系统")
        self.showFullScreen()
        
        # 设置窗口背景
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FAFAFA;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Header 区域 - 更美观的设计
        header = QWidget()
        header.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF9800, stop:1 #FFA726);
                border-bottom: 3px solid #F57C00;
            }
        """)
        header.setFixedHeight(70)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 10, 20, 10)

        # Logo 区域
        logo_container = QWidget()
        logo_container.setFixedSize(50, 50)
        logo_container.setStyleSheet("""
            background-color: white;
            border-radius: 25px;
            border: 3px solid #FFB74D;
        """)
        # logo_label = QLabel("🍠")
        # logo_label.setAlignment(Qt.AlignCenter)
        # logo_label.setFont(QFont("微软雅黑", 24))
        # logo_label.setStyleSheet("background-color: transparent; border: none;")
        # logo_layout = QHBoxLayout(logo_container)
        # logo_layout.setContentsMargins(0, 0, 0, 0)
        # logo_layout.addWidget(logo_label)

        # 标题
        title_label = QLabel("甘薯知识助手")
        title_label.setFont(QFont("微软雅黑", 22, QFont.Bold))
        title_label.setAlignment(Qt.AlignVCenter)
        title_label.setStyleSheet("""
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            background-color: transparent;
            border: none;
        """)

        # 用户信息
        user_container = QWidget()
        user_container.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 5px 15px;
        """)
        user_label = QLabel(f"{self.user_name}")
        user_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        user_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        user_label.setStyleSheet("""
            color: white;
            background-color: transparent;
            border: none;
        """)
        user_layout = QHBoxLayout(user_container)
        user_layout.setContentsMargins(10, 5, 10, 5)
        user_layout.addWidget(user_label)

        header_layout.addWidget(logo_container)
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(user_container)

        main_layout.addWidget(header)
        main_layout.addWidget(self.status_indicator)
        main_layout.addWidget(self.chat_area, 1)

        # ESC 退出
        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.close)

    def setup_asyncio_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_asyncio_events)
        self.timer.start(10)

    def _process_asyncio_events(self):
        self.loop.call_soon(lambda: None)
        self.loop.stop()
        self.loop.run_forever()

    def add_task(self, coro):
        task = self.loop.create_task(coro)
        self.current_tasks.append(task)
        task.add_done_callback(lambda t: self.current_tasks.remove(t) if t in self.current_tasks else None)
        return task

    async def play_welcome_and_listen(self):
        welcome_msg = f"您好，{self.user_name}！我是甘薯知识助手，请通过语音向我提问关于甘薯的问题。"
        # 显示文字
        self.chat_area.add_message(welcome_msg)
        # 切到"回答中"状态
        self.status_indicator.set_answering()
        # 播报并等待完成
        await self.tts_streamer.speak_text(welcome_msg, wait=True)
        
        # 语音提示（不显示在屏幕上）
        await self.tts_streamer.speak_text("您有什么想问的吗？", wait=True)
        
        # 设置为"已回答"状态
        self.status_indicator.set_answerd()
        # 切到"聆听"状态并启动连续聆听
        self.status_indicator.set_listening()
        await asyncio.sleep(0.2)
        self.add_task(self.continuous_listening_task())

    async def continuous_listening_task(self):
        while True:
            try:
                # 保证 TTS 完毕
                if self.tts_streamer.is_speaking:
                    await self.tts_streamer.wait_until_done()
                    await asyncio.sleep(0.1)
                await self.clear_audio_buffer()

                # 语音识别
                text = await self.asr_helper.real_time_recognition(
                    callback=lambda status: self.bridge.status_changed.emit(status)
                )

                if text and not self.is_processing:
                    self.is_processing = True
                    # 新问题，切到"处理"状态
                    self.bridge.add_user_message.emit(text)
                    self.status_indicator.set_processing()

                    # 开始机器人消息
                    self.bridge.start_bot_message.emit()
                    self.current_answer = ""
                    
                    # 文本缓冲区
                    text_buffer = ""
                    # 计算缓冲区中标点符号的数量
                    punctuation_count = 0
                    # 设置标点符号阈值，达到这个数量才发送
                    punctuation_threshold = 3  # 可以调整为3或4
                    
                    # 设置为回答状态
                    self.status_indicator.set_answerd()

                    # 流式生成回答并同步进行语音合成
                    async for chunk in self.qa_model.ask_stream(text):
                        self.current_answer += chunk
                        self.bridge.update_bot_message.emit(self.current_answer)
                        
                        # 将新块添加到缓冲区
                        text_buffer += chunk
                        
                        # 计算当前块中的标点符号数量
                        new_punctuations = len(re.findall(r'[。，,.!?！？;；]', chunk))
                        punctuation_count += new_punctuations
                        
                        # 条件：达到标点符号阈值或缓冲区足够长
                        if (punctuation_count >= punctuation_threshold and len(text_buffer) >= 15) or len(text_buffer) > 80:
                            if text_buffer.strip():
                                await self.tts_streamer.speak_text(text_buffer, wait=False)
                            
                            # 重置缓冲区和计数器
                            text_buffer = ""
                            punctuation_count = 0
                        
                        # 给UI渲染的时间
                        await asyncio.sleep(0.01)
                    
                    # 处理剩余的文本缓冲区
                    if text_buffer.strip():
                        await self.tts_streamer.speak_text(text_buffer, wait=False)
                    
                    # 等待所有语音播放完成
                    await self.tts_streamer.wait_until_done()
                    
                    # 语音提示继续对话（不显示在屏幕上）
                    follow_up = random.choice(self.follow_up_prompts)
                    await self.tts_streamer.speak_text(follow_up, wait=True)
                    
                    # 播报结束，切到"聆听"
                    self.status_indicator.set_listening()
                    self.is_processing = False

                await asyncio.sleep(0.5)
            except Exception as e:
                logging.error(f"连续聆听过程中出错: {e}")
                self.is_processing = False
                await asyncio.sleep(1)

    async def clear_audio_buffer(self):
        try:
            if hasattr(self.asr_helper, 'stream') and self.asr_helper.stream:
                await asyncio.sleep(0.1)
                while self.asr_helper.stream.get_read_available() > 0:
                    self.asr_helper.stream.read(self.asr_helper.CHUNK, exception_on_overflow=False)
                logging.info("音频缓冲区已清理")
        except Exception as e:
            logging.warning(f"清理音频缓冲区时出错: {e}")

    def update_status(self, status):
        if status == "waiting":
            self.status_indicator.set_waiting()
        elif status == "listening":
            self.status_indicator.set_listening()
        elif status == "processing":
            self.status_indicator.set_processing()
        elif status == "answering":
            self.status_indicator.set_answerd()

    def add_question(self, text):
        self.chat_area.add_message(text, is_user=True)
        self.status_indicator.set_processing()

    def start_bot_message(self):
        self.current_bot_bubble = self.chat_area.add_message("", is_user=False)
        # 在 start_bot_message 中加动画控制
        self.loading_dots_timer = QTimer()
        self.loading_dots = ""
        self.loading_dots_timer.timeout.connect(self.animate_loading_dots)
        self.loading_dots_timer.start(500)  # 每 500ms 更新一次

    def animate_loading_dots(self):
        self.loading_dots = "." * ((len(self.loading_dots) % 3) + 1)
        if self.current_bot_bubble:
            self.current_bot_bubble.update_text(f"正在思考中{self.loading_dots}")
    def update_bot_message(self, text):
        """更新机器人消息"""
        if self.loading_dots_timer.isActive():
            self.loading_dots_timer.stop()
        if self.current_bot_bubble:
            self.current_bot_bubble.update_text(text)
            
            # 使用改进的平滑滚动
            QTimer.singleShot(10, lambda: self.chat_area.smooth_scroll_to_bottom())

    def start_real_time_listening(self):
        if self.is_processing:
            return
        self.status_indicator.set_listening()
        self.add_task(self.continuous_listening_task())

    def stop_recording(self):
        self.asr_helper.stop_recording()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'chat_area') and self.chat_area:
            self.chat_area.update_bubble_widths(self.width())

    def closeEvent(self, event):
        for task in self.current_tasks:
            task.cancel()
        self.asr_helper.close_audio()
        self.timer.stop()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.conversation_manager.save_tracking_data())
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置全局样式
    app.setStyle("Fusion")
    
    window = SweetPotatoGUI("吴家卓")
    window.show()
    
    sys.exit(app.exec_())