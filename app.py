import sys
import asyncio
import time
import re
import json
import random
from datetime import datetime
from collections import deque
import logging
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

# 百度ASR API配置
APP_ID = ''
API_KEY = '' 
SECRET_KEY = ''

# QA模型所需导入
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import nest_asyncio
from openai import OpenAI, AsyncOpenAI
import nest_asyncio
from qwen_agent.agents import Assistant
import random
nest_asyncio.apply()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat.log"), logging.StreamHandler()]
)

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
            context.append(f"qustion: {conv['question']}")
            context.append(f"answer: {conv['answer']}")
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
        self.exit_stack = None
        self.sessions = {}
        self.tools = []

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
        
    # 添加MCP相关功能
    async def connect_to_mcp(self, config_file="mcp_server_config.json"):
        """连接到MCP服务器"""
        try:
            from mcp.server.fastmcp import FastMCP
            from contextlib import AsyncExitStack
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.sse import sse_client
            
            self.exit_stack = AsyncExitStack()
            self.sessions = {}
            self.tools = []
            
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                
            conf = config["mcpServers"]
            for key in conf.keys():
                v = conf[key]
                session = None
                if "url" in v and v['isActive'] and "type" in v and v["type"] == "sse":
                    server_url = v['url']
                    sse_transport = await self.exit_stack.enter_async_context(sse_client(server_url))
                    write, read = sse_transport
                    session = await self.exit_stack.enter_async_context(ClientSession(write, read))
                elif "command" in v and v['isActive']:
                    command = v['command']
                    args = v['args']
                    server_params = StdioServerParameters(command=command, args=args, env=None)
                    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                    stdio1, write1 = stdio_transport
                    session = await self.exit_stack.enter_async_context(ClientSession(stdio1, write1))
                
                if session:
                    await session.initialize()
                    response = await session.list_tools()
                    tools = response.tools
                    for tool in tools:
                        self.sessions[tool.name] = session
                    self.tools += tools
                    
            logging.info("MCP服务已连接")
            return True
        except Exception as e:
            logging.error(f"连接MCP服务失败: {e}")
            return False
            
    async def play_music(self, song_name):
        """使用MCP播放音乐"""
        if not hasattr(self, 'sessions') or 'play_music' not in self.sessions:
            logging.error("MCP音乐服务未连接")
            return "音乐服务未连接"
            
        try:
            result = await self.sessions['play_music'].call_tool("play_music", {"song_name": song_name})
            return result.content[0].text
        except Exception as e:
            logging.error(f"播放音乐失败: {e}")
            return f"播放音乐失败: {str(e)}"
    
    async def stop_music(self):
        """停止音乐播放"""
        if not hasattr(self, 'sessions') or 'stopplay' not in self.sessions:
            return "音乐服务未连接"
            
        try:
            result = await self.sessions['stopplay'].call_tool("stopplay", {})
            return result.content[0].text
        except Exception as e:
            logging.error(f"停止音乐失败: {e}")
            return f"停止音乐失败: {str(e)}"

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
# 知识问答类 (扩展自qa_model_easy.py)
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
        embedding_model_path="/home/wuye/vscode/raspberrypi_5/rasoberry/text2vec_base_chinese_q8.gguf",
        conversation_manager=None,
        model_name="qwen-turbo-latest",
        api_key='',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        mcp_config_path="mcp_server_config.json"
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
        
        # 初始化对话管理器
        self.conversation_manager = conversation_manager or ConversationManager()
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url 
        
        # 初始化同步和异步的OpenAI客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # 系统消息设置
        self.sys_msg = {
            "role": "system",                                                           
            "content": "回答简洁"
        }
        
        # MCP配置初始化
        self.mcp_config_path = mcp_config_path
        self.load_mcp_config()

        # 仅保留最基本的Agent配置
        self.llm_cfg = {
            'model': self.model_name,
            'model_server': 'dashscope',
            'api_key': self.api_key,
            'generate_cfg': {
                'top_p': 0.8,
                'thought_in_content': False,
                'max_tokens': 400
            }
        }
        
        system_instruction = '''/no think'''
        self.bot = Assistant(llm=self.llm_cfg,
                          system_message=system_instruction,
                          function_list=[self.config])
        
        # 添加音乐相关关键词分类
        self.music_commands = {
            "播放": ["播放", "放一首", "来一首", "听听", "我想听"],
            "暂停": ["暂停", "停止", "先停下"],
            "继续": ["继续", "恢复", "接着放","再来"],
            "停止": ["停止", "关闭音乐", "我不听了"],
            "下一首": ["下一首", "换一首", "播放下一首"],
            "播放列表": ["播放列表", "歌单", "列表", "有什么歌"]
        }
        
        # 添加搜索相关关键词
        self.search_keywords = [
            "搜索", "查找", "查询", "搜一下", "查一下", 
            "查找", "搜索", "搜", "搜一搜","日历","帮我",
            "帮我查", "帮我搜", "请搜索", "请查找"
        ]
        
        # 添加网络信息相关关键词（需要实时信息的查询）
        self.web_info_keywords = [
            "最新", "最近", "现在", "今天", "目前", "当前",
            "实时", "新闻", "热点", "天气", "股价", "比分",
            "排行", "趋势", "动态", "更新", "价格","明天","后天","昨天","前天","大后天","大前天","前几天","后几天","之后","你知道吗"
        ,"月","号","年","天","那天","几点","点钟"
        ]

    def _load_vectorstore_with_retry(self, max_retries=3):
        for i in range(max_retries):
            try:
                return FAISS.load_local(self.faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
            except Exception as e:
                logging.warning(f"第{i+1}次加载 FAISS 失败: {e}")
                time.sleep(1)
        raise RuntimeError("加载向量存储失败")
    
    def load_mcp_config(self):
        """加载MCP服务器配置"""
        try:
            with open(self.mcp_config_path, "r") as f:
                self.config = json.load(f)
                logging.info(f"已加载MCP配置: {self.config}")
        except Exception as e:
            logging.error(f"加载MCP配置失败: {e}")
            self.config = {"mcpServers": {}}
    
    async def call_tool(self, tool_name, tool_args):
        """使用Qwen Agent调用MCP工具"""
        try:
     
            tool_args_str = json.dumps(tool_args, ensure_ascii=False)

 
            result = await asyncio.to_thread(self.bot._call_tool, tool_name, tool_args_str)
            
    
            return result
        except Exception as e:
            logging.error(f"调用工具 {tool_name} 失败: {e}")
            return {"error": f"调用工具失败: {str(e)}"}
    
    def detect_music_intent(self, question):
        """检测音乐相关意图和具体命令"""
        question_lower = question.lower()
        
        # 首先检查特定的音乐控制命令（优先级更高）
        specific_commands = {
            "播放列表": ["播放列表", "显示播放列表", "当前播放列表", "歌单", "列表","我想听"],
            "下一首": ["下一首", "换一首", "下首歌", "播放下一首","换一首歌"],
            "上一首": ["上一首", "前一首", "播放上一首"],
            "暂停": ["暂停", "停下", "先停"],
            "继续": ["继续", "恢复", "接着放"],
            "停止": ["停止播放", "关闭音乐", "不听了", "停止"]
        }
        
        # 检查特定命令
        for command, keywords in specific_commands.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return {"command": command}
        
        # 然后检查播放相关命令（使用更精确的匹配）
        play_patterns = [
            (r"播放\s*(.+)", "播放"),
            (r"放一首\s*(.+)", "播放"),
            (r"来一首\s*(.+)", "播放"),
            (r"听听\s*(.+)", "播放"),
            (r"我想听\s*(.+)", "播放"),
            (r"点一首\s*(.+)","播放")
        ]
        
        for pattern, command in play_patterns:
            match = re.search(pattern, question)
            if match:
                song_name = match.group(1).strip()
                # 过滤掉可能被误识别的词汇
                if song_name and song_name not in ["下一首", "上一首", "列表", "播放列表"]:
                    return {"command": command, "song_name": song_name}
        
        return None
    
    def detect_search_intent(self, question):
        """检测搜索相关意图"""
        question_lower = question.lower()
        
        # 直接的搜索命令检测
        for keyword in self.search_keywords:
            if keyword in question_lower:
                # 提取搜索内容
                patterns = [
                    f"{keyword}(.+)",
                    f"请{keyword}(.+)",
                    f"帮我{keyword}(.+)",
                    f"(.+){keyword}"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, question_lower)
                    if match:
                        search_query = match.group(1).strip()
                        # 清理查询词
                        for kw in self.search_keywords:
                            search_query = search_query.replace(kw, "").strip()
                        
                        if search_query:
                            return {"command": "search", "query": search_query}
                return {"command": "search", "query": question}
        
        # 检测是否需要网络实时信息
        for keyword in self.web_info_keywords:
            if keyword in question_lower:
                return {"command": "search", "query": question}
        
        # 检测特定的网络查询模式
        web_patterns = [
            r"(.+)是什么",
            r"什么是(.+)",
            r"(.+)怎么样",
            r"(.+)的价格",
            r"(.+)新闻",
            r"(.+)最新消息"
        ]
        
        for pattern in web_patterns:
            match = re.search(pattern, question)
            if match:
                # 检查是否包含网络相关词汇
                if any(keyword in question_lower for keyword in self.web_info_keywords):
                    return {"command": "search", "query": question}
        
        return None
    
    async def handle_search_command(self, intent):
        """处理搜索相关命令"""
        command = intent.get("command")
        query = intent.get("query", "")
        
        if command == "search" and query:
            # 调用正确的搜索工具名称
            tool_result = await self.call_tool("web_search-web_search", {"query": query, "limit": 5})
            
            try:
                # 解析搜索结果
                if isinstance(tool_result, str):
                    search_data = json.loads(tool_result)
                else:
                    search_data = tool_result
                
                # 检查搜索状态
                if search_data.get("status") == "error":
                    return f"搜索失败：{search_data.get('message', '未知错误')}"
                
                # 处理成功的搜索结果
                if search_data.get("status") == "success" and "results" in search_data:
                    results = search_data["results"]
                    if results:
                        # 如果只有一个结果，直接返回内容
                        if len(results) == 1:
                            return results[0].get("content", "搜索结果为空")
                        
                        # 多个结果时格式化输出
                        response = f"为您搜索到以下关于({query})的信息：\n\n"
                        for i, result in enumerate(results[:3], 1):
                            content = result.get("content", "")
                            if content:
                                response += f"{i}. {content}\n\n"
                        
                        return response.strip()
                    else:
                        return f"搜索({query})未找到相关结果。"
                
                # 如果返回格式不符合预期，尝试直接返回
                return str(search_data)
                    
            except Exception as e:
                logging.error(f"处理搜索结果失败: {e}")
                return f"搜索({query})时出现错误：{str(e)}"
        
        return "请提供要搜索的内容。"
    
    async def handle_music_command(self, intent): 
        """处理音乐相关命令"""
        command = intent.get("command")
        tool_result = None 

        if command == "播放":
            song_name = intent.get("song_name", "")
            if song_name:
                
                tool_result = await self.call_tool("netease_music-play_music", {"song_name": song_name})
                if isinstance(tool_result, dict) and "status" in tool_result:
                    return tool_result["status"]
               
                logging.warning(f"播放音乐调用结果异常: {tool_result}")
                return str(tool_result) if tool_result else f"尝试播放 {song_name} 时出错，未收到明确结果。"
            else:
                return "请告诉我您想听的歌曲名称或歌手"

        elif command == "停止":
           
            tool_result = await self.call_tool("netease_music-stopplay", {})

        elif command == "暂停":
        
            tool_result = await self.call_tool("netease_music-pauseplay", {})

        elif command == "继续":
            
            tool_result = await self.call_tool("netease_music-unpauseplay", {})

        elif command == "下一首":
           
            tool_result = await self.call_tool("netease_music-next_song", {})
            if isinstance(tool_result, dict) and "status" in tool_result:
                return tool_result["status"]
     

        elif command == "播放列表":

            tool_result = await self.call_tool("netease_music-get_playlist", {})

        else:
            return "无法处理该音乐指令。"

       
        if tool_result is not None:
            return str(tool_result)

        
        return "音乐操作执行完毕，但未收到明确结果。"

    
    async def ask_stream(self, question, use_context=True, use_tools=True):
        """使用流式响应回答问题"""
        start_time = time.time()
        
        try:
            # 检测意图并处理特殊命令
            if use_tools:
                # 检测音乐命令
                music_intent = self.detect_music_intent(question)
                if music_intent:
                    result = await self.handle_music_command(music_intent)
                    yield result
                    
                    # 记录对话
                    response_time = time.time() - start_time
                    await self.conversation_manager.add_conversation_entry(question, result, response_time)
                    await self.conversation_manager.save_tracking_data()
                    return
                
                # 检测搜索命令
                search_intent = self.detect_search_intent(question)
                if search_intent:
                    # 先返回搜索提示
                    yield "正在执行网络搜索任务..."
                    
                    # 执行搜索
                    result = await self.handle_search_command(search_intent)
                    
                    # 返回搜索结果
                    yield result
                    
                    # 记录对话
                    response_time = time.time() - start_time
                    await self.conversation_manager.add_conversation_entry(question, result, response_time)
                    await self.conversation_manager.save_tracking_data()
                    return
            
            # 非特殊指令处理 - 使用知识库回答
            context = ""
            if use_context:
                context = self.conversation_manager.get_conversation_context(max_context=3)
            
            # 获取相关文档
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

            # 构建查询提示
            query = "你是一个甘薯专家，请你以说话的标准回答,"
            
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
                # 使用与test.py相同的参数和格式
                stream = self.client.chat.completions.create(
                    model=self.model_name,  
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
                
                # 记录对话
                response_time = time.time() - start_time
                await self.conversation_manager.add_conversation_entry(question, full_response, response_time)
                await self.conversation_manager.save_tracking_data()
                        
            except Exception as e:
                error_msg = f"API调用出错: {e}"
                logging.error(f"API调用失败: {e}")
                await self.conversation_manager.record_error("API_ERROR", str(e))
                yield error_msg
                
        except Exception as e:
            error_msg = f"处理问题时出错: {e}"
            logging.error(f"处理问题失败: {e}")
            await self.conversation_manager.record_error("PROCESS_ERROR", str(e))
            yield error_msg
            
    def get_player_status(self):
        """获取音乐播放器状态"""
        try:
            # self.bot._call_tool 是同步的, 需要在异步代码中用 asyncio.to_thread
            # 但这里是同步方法，所以直接调用是OK的。
            # 返回的可能是 Observation 对象
            return self.bot._call_tool("netease_music-isPlaying", "{}")
        except Exception as e:
            logging.error(f"获取播放器状态失败: {e}")
            return "not playing" # 或者返回一个表示错误的 Observation 结构

# ==================================
# UI 组件类
# ==================================
class MessageBubble(QWidget):
    """优化的聊天气泡组件 - 高级UI设计"""
    def __init__(self, text, is_user=False, parent=None):
        super().__init__(parent)
        self.text = text
        self.is_user = is_user
        self._msg_label = None

        # 头像路径
        self.avatar_path = "guzz.png"
        self.robot_path = "sweetpotato.jpg"

        # 为7寸屏幕优化布局 - 增强设计
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 15, 0, 15)  
        layout.setSpacing(18)

        avatar_size = 68
        avatar_label = QLabel()
        avatar_label.setFixedSize(avatar_size, avatar_size)
        avatar_label.setAlignment(Qt.AlignCenter)
        
        # 增强头像设计 - 添加阴影效果
        avatar_label.setStyleSheet(f"""
            border-radius: 8px;
            background-color: #F5F5F5;
            border: 4px solid {"#4CAF50" if is_user else "#FF9800"};
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
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
        self._msg_label.setFont(QFont("微软雅黑", 17, QFont.Normal))  # 增大字体
        self._msg_label.setWordWrap(True)
        self._msg_label.setMaximumWidth(820)  # 增加最大宽度
        
        # 增强气泡设计 - 添加渐变和阴影
        msg_style = f"""
            background: {"qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #E8F5E9, stop:1 #C8E6C9)" if is_user else "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #F8F8F8)"};
            color: #2E2E2E;
            border-radius: 22px;
            padding: 18px 24px;
            border: 2px solid {"#81C784" if is_user else "#E0E0E0"};
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin: 5px;
        """
        self._msg_label.setStyleSheet(msg_style)

        # 按消息来源设置左右布局
        if is_user:
            layout.addStretch()
            layout.addWidget(self._msg_label)
            layout.addWidget(avatar_label)
        else:
            layout.addWidget(avatar_label)
            layout.addWidget(self._msg_label)
            layout.addStretch()

        self.setMinimumHeight(85)

    @property
    def msg_label(self):
        return self._msg_label

    def update_text(self, text):
        if self._msg_label:
            self._msg_label.setText(text)


class ChatArea(QScrollArea):
    """优化的聊天区域 - 高级UI设计"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置滚动区域属性
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # 美化滚动条样式 - 现代化设计
        self.setStyleSheet("""
            QScrollArea {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8FBF8, stop:1 #F0F8F0);
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: rgba(255, 255, 255, 0.5);
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(76, 175, 80, 0.6), stop:1 rgba(76, 175, 80, 0.8));
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(76, 175, 80, 0.8), stop:1 rgba(76, 175, 80, 1.0));
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # 创建容器小部件
        self.container = QWidget()
        self.container.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F8FBF8, stop:1 #F0F8F0);
        """)
        
        # 创建垂直布局
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(25)  # 增加间距使界面更清爽
        self.layout.setContentsMargins(25, 25, 25, 25)
        
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
        max_width = min(820, int(width * 0.7))  
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            if item and item.widget():
                bubble = item.widget()
                if hasattr(bubble, 'msg_label'):
                    bubble.msg_label.setMaximumWidth(max_width)

class StatusIndicator(QWidget):
    """语音状态指示器 - 高级UI设计"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FFFFFF, stop:1 #F8F8F8);
                border-bottom: 3px solid #E8F5E9;
            }
        """)
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(25, 12, 25, 12)
        
        # 状态图标 - 增强设计
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(28, 28)
        self.icon_label.setStyleSheet("""
            background-color: #5B89DB; 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        """)
        
        # 状态文本 - 增强字体
        self.text_label = QLabel("正在初始化...")
        self.text_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        self.text_label.setStyleSheet("color: #2E2E2E; text-shadow: 0 1px 3px rgba(0,0,0,0.1);")
        
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.text_label)
        self.layout.addStretch()
        
        # 初始状态
        self.set_waiting()
        
    def set_waiting(self):
        """设置等待状态"""
        self.text_label.setText("等待语音输入...")
        self.icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #5B89DB, stop:1 #7BA5E8); 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        """)
        
    def set_listening(self):
        """设置监听状态"""
        self.text_label.setText("正在聆听...")
        self.icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #F44336, stop:1 #E57373); 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        """)
        
    def set_processing(self):
        """设置处理状态"""
        self.text_label.setText("正在思考...")
        self.icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #FFC107, stop:1 #FFD54F); 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        """)
        
    def set_answering(self):
        '''设置回答状态'''
        self.text_label.setText("正在播放欢迎语...")
        self.icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #E91E63, stop:1 #F06292); 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        """)
        
    def set_searching(self):
        """设置搜索状态"""
        self.text_label.setText("正在进行网络搜索...")
        self.icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #673AB7, stop:1 #9575CD); 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        """)
        
    def set_playing_music(self):
        """设置音乐播放状态"""
        self.text_label.setText("正在播放音乐...")
        self.icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #009688, stop:1 #4DB6AC); 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        """)
        
    def set_music_processing(self):
        """设置音乐处理状态"""
        self.text_label.setText("正在处理音乐请求...")
        self.icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #795548, stop:1 #A1887F); 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        """)
        
    def set_music_listening(self):
        """设置音乐播放时的监听状态"""
        self.text_label.setText("播放音乐中，正在聆听...")
        self.icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #03A9F4, stop:1 #29B6F6); 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        """)
        
    def set_music_thinking(self):
        """设置音乐播放时的思考状态"""
        self.text_label.setText("播放音乐中，正在思考...")
        self.icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #607D8B, stop:1 #78909C); 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        """)
        
    def set_answerd(self):
        """设置回答状态"""
        self.text_label.setText("正在回答中...")
        self.icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #4CAF50, stop:1 #66BB6A); 
            border-radius: 14px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
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
        
        # 修改音乐交互模式，简化为两种模式
        self.music_interaction_mode = "normal"  # normal, music_mode
        self.music_listen_task = None
        self.is_searching = False 
        
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
        self.current_question_start_time = None
        self.first_interaction = True

        # UI 与事件循环
        self.init_ui()
        self.setup_asyncio_event_loop()
        
        # MCP 初始化
        self.add_task(self.initialize_mcp())
        
        # 播放欢迎并开始流程
        self.add_task(self.play_welcome_and_listen())

    async def initialize_mcp(self):
        """初始化MCP服务"""
        self.status_indicator.text_label.setText("正在初始化MCP服务...")
        self.mcp_connected = await self.tts_streamer.connect_to_mcp()
        
        if self.mcp_connected:
            logging.info("✅ MCP服务已成功连接")
        else:
            logging.warning("⚠️ MCP服务连接失败，部分功能可能不可用")

    def init_ui(self):
        """初始化UI - 高级设计"""
        self.setWindowTitle("甘薯知识问答系统")
        self.showFullScreen()
        
        # 设置窗口背景
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8FBF8, stop:1 #F0F8F0);
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
                    stop:0 #FF9800, stop:1 #FFB74D);
                border-bottom: 4px solid #F57C00;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }
        """)
        header.setFixedHeight(80)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(25, 12, 25, 12)

        # Logo 区域 - 增强设计
        logo_container = QWidget()
        logo_container.setFixedSize(56, 56)
        logo_container.setStyleSheet("""
            background-color: white;
            border-radius: 28px;
            border: 4px solid #FFB74D;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        """)

        # 标题 - 增强字体
        title_label = QLabel("甘薯知识助手")
        title_label.setFont(QFont("微软雅黑", 26, QFont.Bold))
        title_label.setAlignment(Qt.AlignVCenter)
        title_label.setStyleSheet("""
            color: white;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
            background-color: transparent;
            border: none;
        """)

        # 用户信息 - 增强设计
        user_container = QWidget()
        user_container.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.25);
            border-radius: 22px;
            padding: 8px 18px;
            border: 2px solid rgba(255, 255, 255, 0.3);
        """)
        user_label = QLabel(f"{self.user_name}")
        user_label.setFont(QFont("微软雅黑", 18, QFont.Bold))
        user_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        user_label.setStyleSheet("""
            color: white;
            background-color: transparent;
            border: none;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        """)
        user_layout = QHBoxLayout(user_container)
        user_layout.setContentsMargins(12, 8, 12, 8)
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
    
    def set_search_status(self):
        """设置正确的搜索状态"""
        self.status_indicator.set_searching()
        self.is_searching = True
        
    def clear_search_status(self):
        """清除搜索状态并返回到适当的状态"""
        self.is_searching = False
        if self.music_interaction_mode == "music_mode":
            self.status_indicator.set_music_listening()
        else:
            self.status_indicator.set_listening()

    async def play_welcome_and_listen(self):
        welcome_msg = f"您好，{self.user_name}！我是甘薯知识助手，请通过语音向我提问关于甘薯的问题。"
        # 显示文字
        self.chat_area.add_message(welcome_msg)
        # 切到"回答中"状态
        self.status_indicator.set_answering()
        # 播报并等待完成
        await self.tts_streamer.speak_text(welcome_msg, wait=True)
        
        # 设置为"已回答"状态
        self.status_indicator.set_answerd()
        # 切到"聆听"状态并启动连续聆听
        self.status_indicator.set_listening()
        await asyncio.sleep(0.2)
        self.add_task(self.continuous_listening_task())

    async def handle_music_interaction(self, text, music_intent):
        """处理音乐相关的交互逻辑 - 简化版本，参考main_stream.py"""
        # 设置状态为音乐处理
        self.status_indicator.set_music_processing()
        
        # 使用qa模型处理音乐指令
        result = await self.qa_model.handle_music_command(music_intent)
        
        # 添加交互显示
        self.bridge.add_user_message.emit(text)
        self.bridge.start_bot_message.emit()
        self.bridge.update_bot_message.emit(result)
        
        # 记录对话
        response_time = time.time() - self.current_question_start_time
        question = music_intent.get("song_name", "音乐操作")
        await self.conversation_manager.add_conversation_entry(question, result, response_time)
        await self.conversation_manager.save_tracking_data()
        
        # 处理播放命令 - 进入音乐模式
        if music_intent.get("command") == "播放":
            # 播放命令结果
            if result:
                clean_result = result.replace("11", "").strip()
                await self.tts_streamer.speak_text(clean_result, wait=True)
            
            # 进入音乐模式
            self.music_interaction_mode = "music_mode"
            logging.info("🎵 已进入音乐模式，将无间隙持续监听语音命令")
            
            # 启动音乐监听任务
            self.music_listen_task = self.add_task(self.music_mode_listening())
            self.status_indicator.set_music_listening()
            
        # 处理播放列表命令 - 只打印不读出
        elif music_intent.get("command") == "播放列表":
            if result:
                clean_result = result.replace("11", "").strip()
                # 只打印到控制台，不进行语音播报
                print(f"\n📋 当前播放列表:\n{clean_result}")
                # 简短提示已显示播放列表
                await self.tts_streamer.speak_text("播放列表已显示", wait=True)
                self.status_indicator.set_listening()
        
        # 处理暂停命令 - 暂停后自动进入问答模式
        elif music_intent.get("command") == "暂停":
            # 播放操作结果
            if result:
                clean_result = result.replace("11", "").strip()
                await self.tts_streamer.speak_text(clean_result, wait=True)
            
            # 音乐已暂停，切换到普通问答模式
            self.music_interaction_mode = "normal"
            logging.info("🎵 音乐已暂停，切换到问答模式")
            self.status_indicator.set_listening()
        
        # 处理继续播放命令 - 从问答模式返回音乐模式
        elif music_intent.get("command") == "继续":
            # 播放操作结果
            if result:
                clean_result = result.replace("11", "").strip()
                await self.tts_streamer.speak_text(clean_result, wait=True)
            
            # 重新进入音乐模式
            self.music_interaction_mode = "music_mode"
            logging.info("🎵 音乐继续播放，重新进入音乐模式")
            
            # 启动音乐监听任务
            self.music_listen_task = self.add_task(self.music_mode_listening())
            self.status_indicator.set_music_listening()
        
        # 其他音乐命令 - 仅播放结果，保持当前模式
        else:
            if result:
                clean_result = result.replace("11", "").strip()
                await self.tts_streamer.speak_text(clean_result, wait=True)
            
            # 根据当前模式设置状态
            if self.music_interaction_mode == "music_mode":
                self.status_indicator.set_music_listening()
            else:
                self.status_indicator.set_listening()
        
        return True
    
    async def music_mode_listening(self):
        """音乐模式：无间隙持续监听用户指令 - 参考main_stream.py的实现"""
        try:
            logging.info("🎵 开始无间隙音乐模式监听")
            
            while self.music_interaction_mode == "music_mode":
                # 检查播放器状态
                player_status = self.qa_model.get_player_status()
                player_status_str = ""
                if hasattr(player_status, 'content') and player_status.content and isinstance(player_status.content[0].get('text'), str):
                     player_status_str = player_status.content[0]['text']
                elif isinstance(player_status, str):
                     player_status_str = player_status
                
                if player_status_str == "stopped" or player_status_str == "not playing":
                    # 音乐播放完毕，自动退出音乐模式
                    logging.info("🎵 音乐播放已结束，退出音乐模式")
                    self.music_interaction_mode = "normal"
                    await self.tts_streamer.speak_text("音乐播放已结束。", wait=True)
                    self.status_indicator.set_listening()
                    break
                
                # 直接开始语音识别 - 不清理缓冲区
                # 这确保我们始终在监听，无盲区
                logging.info("🎵 持续监听音乐命令中...")
                self.status_indicator.set_music_listening()
                
                command_result = await self.asr_helper.real_time_recognition(
                    callback=lambda status: self.bridge.status_changed.emit(status)
                )
                
                # 处理任何检测到的命令
                if command_result and command_result.strip():
                    logging.info(f"🎵 音乐模式中检测到指令: {command_result}")
                    
                    # 检查是否为音乐相关指令
                    music_intent = self.qa_model.detect_music_intent(command_result)
                    if music_intent:
                        # 记录开始时间
                        self.current_question_start_time = time.time()
                        self.status_indicator.set_music_thinking()
                        
                        # 处理音乐指令
                        result = await self.qa_model.handle_music_command(music_intent)
                        
                        # 添加到对话界面并显示在屏幕上
                        self.bridge.add_user_message.emit(command_result)
                        self.bridge.start_bot_message.emit()
                        self.bridge.update_bot_message.emit(result)
                        
                        # 在控制台也打印出来
                        print(f"\n🎵 音乐指令: {command_result}")
                        print(f"🎵 执行结果: {result}")
                        
                        # 记录对话
                        response_time = time.time() - self.current_question_start_time
                        question = music_intent.get("song_name", "音乐操作")
                        await self.conversation_manager.add_conversation_entry(question, result, response_time)
                        await self.conversation_manager.save_tracking_data()
                        
                        # 播放操作结果
                        if result:
                            clean_result = result.replace("11", "").strip()
                            await self.tts_streamer.speak_text(clean_result, wait=True)
                        
                        # 特殊命令处理
                        if music_intent.get("command") in ["暂停", "停止", "退出"]:
                            # 退出音乐模式
                            self.music_interaction_mode = "normal"
                            logging.info(f"🎵 由于{music_intent.get('command')}命令退出音乐模式")
                            self.status_indicator.set_listening()
                            break
                    else:
                        # 非音乐命令，忽略处理，只记录日志
                        logging.info(f"🎵 在音乐模式中检测到非音乐命令，忽略处理: {command_result}")
                        print(f"🎵 忽略非音乐指令: {command_result}")
                        # 继续监听，不作任何处理
                
                # 不在识别循环之间添加任何延迟，但给出微小的让权时间
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            logging.info("🎵 音乐监听任务被取消")
        except Exception as e:
            logging.error(f"🎵 音乐监听任务出错: {e}")
            # 发生错误时恢复到正常模式
            self.music_interaction_mode = "normal"
            self.status_indicator.set_listening()

    async def clear_audio_buffer(self):
        try:
            if hasattr(self.asr_helper, 'stream') and self.asr_helper.stream:
                await asyncio.sleep(0.1)
                while self.asr_helper.stream.get_read_available() > 0:
                    self.asr_helper.stream.read(self.asr_helper.CHUNK, exception_on_overflow=False)
                logging.info("音频缓冲区已清理")
        except Exception as e:
            logging.warning(f"清理音频缓冲区时出错: {e}")

    async def continuous_listening_task(self):
        while True:
            try:
                # 保证 TTS 完毕
                if self.tts_streamer.is_speaking:
                    await self.tts_streamer.wait_until_done()
                    await asyncio.sleep(0.1) # 确保TTS流完全结束后有短暂喘息
                await self.clear_audio_buffer()
                
                # 如果在音乐模式，则由音乐监听任务处理
                if self.music_interaction_mode == "music_mode":
                    # 检查音乐监听任务是否运行
                    if not hasattr(self, 'music_listen_task') or self.music_listen_task is None or self.music_listen_task.done():
                        self.music_listen_task = self.add_task(self.music_mode_listening())
                        
                    # 短暂等待后再检查状态
                    await asyncio.sleep(0.5)
                    continue

                # 正常模式下的问答流程
                if self.music_interaction_mode == "normal" and not self.is_processing:
                    prompt_text = f"您好，{self.user_name}！我是甘薯知识助手。" if self.first_interaction else random.choice(self.follow_up_prompts)
                    if self.first_interaction:
                        self.first_interaction = False

                    try:
                        await self.tts_streamer.speak_text(prompt_text, wait=True)
                    except Exception as e:
                        logging.error(f"⚠️ 语音提示失败: {e}")
                    
                    await asyncio.sleep(0.3)
                    await self.clear_audio_buffer()
                    
                    # 开始语音识别
                    self.status_indicator.set_listening()
                    text = await self.asr_helper.real_time_recognition(
                        callback=lambda status: self.bridge.status_changed.emit(status)
                    )
                    
                    # 检查语音识别结果
                    if not text or text.strip() == "" or text.lower() in ["嗯。", "嗯嗯。", "嗯嗯嗯。", "啊。", "啊？"] or re.fullmatch(r"嗯+", text.lower()):
                        logging.info(f"❌ 未检测到有效语音输入或输入为无意义词: '{text}'")
                        continue
                    
                    # 处理有效输入
                    self.is_processing = True
                    self.current_question_start_time = time.time()
                    
                    # 检查退出命令
                    if any(word in text.lower() for word in ["拜拜", "再见", "退出"]):
                        logging.info(f"🚪 收到退出命令: '{text}'")
                        self.bridge.add_user_message.emit(text)
                        self.bridge.start_bot_message.emit()
                        self.bridge.update_bot_message.emit("再见！感谢使用甘薯知识助手。")
                        
                        # 取消音乐监听任务
                        if hasattr(self, 'music_listen_task') and self.music_listen_task and not self.music_listen_task.done():
                            self.music_listen_task.cancel()
                        
                        await self.tts_streamer.speak_text("好的，感谢使用甘薯知识助手，再见！", wait=True)
                        self.close()
                        return
                    
                    # 检测音乐命令
                    music_intent = self.qa_model.detect_music_intent(text)
                    if music_intent:
                        await self.handle_music_interaction(text, music_intent)
                        self.is_processing = False
                        continue
                    
                    # 检测搜索命令
                    search_intent = self.qa_model.detect_search_intent(text)
                    if search_intent:
                        self.is_searching = True
                        self.bridge.add_user_message.emit(text)
                        self.status_indicator.set_searching()
                        self.bridge.start_bot_message.emit()
                        self.bridge.update_bot_message.emit("正在执行网络搜索任务...")
                        
                        result = await self.qa_model.handle_search_command(search_intent)
                        
                        self.is_searching = False
                        self.bridge.update_bot_message.emit(result)
                        
                        # 记录对话
                        response_time = time.time() - self.current_question_start_time
                        await self.conversation_manager.add_conversation_entry(text, result, response_time)
                        await self.conversation_manager.save_tracking_data()
                        
                        # TTS播放搜索结果
                        await self.tts_streamer.speak_text(result, wait=True)
                        
                        self.status_indicator.set_listening()
                        self.is_processing = False
                        continue
                    
                    # 处理普通问题
                    self.bridge.add_user_message.emit(text)
                    self.status_indicator.set_processing()
                    self.bridge.start_bot_message.emit()
                    
                    # 流式处理回答
                    self.current_answer = ""
                    text_buffer = ""
                    punctuation_count = 0
                    punctuation_threshold = 3
                    
                    self.status_indicator.set_answerd()
                    
                    async for chunk in self.qa_model.ask_stream(text):
                        self.current_answer += chunk
                        self.bridge.update_bot_message.emit(self.current_answer)
                        
                        text_buffer += chunk
                        new_punctuations = len(re.findall(r'[。，,.!?！？;；]', chunk))
                        punctuation_count += new_punctuations
                        
                        if (punctuation_count >= punctuation_threshold and len(text_buffer) >= 15) or len(text_buffer) > 80:
                            if text_buffer.strip():
                                await self.tts_streamer.speak_text(text_buffer, wait=False)
                            text_buffer = ""
                            punctuation_count = 0
                        
                        await asyncio.sleep(0.01)
                    
                    # 处理剩余的文本缓冲区
                    if text_buffer.strip():
                        await self.tts_streamer.speak_text(text_buffer, wait=False)
                    
                    # 等待TTS播放完成
                    await self.tts_streamer.wait_until_done()
                    
                    # 记录对话
                    response_time = time.time() - self.current_question_start_time
                    await self.conversation_manager.add_conversation_entry(text, self.current_answer, response_time)
                    await self.conversation_manager.save_tracking_data()
                    
                    self.status_indicator.set_listening()
                    self.is_processing = False
                
                await asyncio.sleep(0.1)  # 添加短暂休眠以减少CPU使用
                
            except asyncio.CancelledError:
                logging.info("连续聆听主循环被取消")
                break
            except Exception as e:
                logging.error(f"连续聆听主循环中出错: {e}", exc_info=True)
                self.is_processing = False
                self.status_indicator.set_waiting()
                await asyncio.sleep(1)
    def update_status(self, status):
        """更新状态指示器"""
        if status == "listening":
            self.status_indicator.set_listening()
        elif status == "processing":
            self.status_indicator.set_processing()
        elif status == "waiting":
            self.status_indicator.set_waiting()
        elif status == "error":
            self.status_indicator.text_label.setText("识别出错，请重试...")
            
        # 处理音乐模式下的状态
        if self.music_interaction_mode == "music_mode":
            if status == "listening":
                self.status_indicator.set_music_listening()
            elif status == "processing":
                self.status_indicator.set_music_thinking()
                
        # 处理搜索状态
        if self.is_searching:
            self.status_indicator.set_searching()
    
    def add_question(self, text):
        """添加用户问题"""
        self.chat_area.add_message(text, is_user=True)
        
    def start_bot_message(self):
        """开始机器人消息"""
        self.current_bot_bubble = self.chat_area.add_message("")
        
    def update_bot_message(self, text):
        """更新机器人消息"""
        if self.current_bot_bubble and self.current_bot_bubble.msg_label:
            self.current_bot_bubble.update_text(text)
            
    def start_real_time_listening(self):
        """启动实时监听"""
        if not self.is_processing:
            self.add_task(self.handle_real_time_listening())
            
    async def handle_real_time_listening(self):
        """处理实时监听"""
        self.is_processing = True
        self.status_indicator.set_listening()
        
        text = await self.asr_helper.real_time_recognition(
            callback=lambda status: self.bridge.status_changed.emit(status)
        )
        
        if text:
            self.bridge.add_user_message.emit(text)
            self.status_indicator.set_processing()
            self.bridge.start_bot_message.emit()
            
            self.current_answer = ""
            async for chunk in self.qa_model.ask_stream(text):
                self.current_answer += chunk
                self.bridge.update_bot_message.emit(self.current_answer)
                await self.tts_streamer.speak_text(chunk, wait=False)
                
            await self.tts_streamer.wait_until_done()
            await self.conversation_manager.add_conversation_entry(text, self.current_answer)
            
        self.is_processing = False
        self.status_indicator.set_waiting()
        
    def resizeEvent(self, event):
        """窗口大小变化时调整界面元素"""
        super().resizeEvent(event)
        # 调整气泡宽度以适应窗口大小
        self.chat_area.update_bubble_widths(event.size().width())
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止所有任务
        for task in self.current_tasks:
            task.cancel()
            
        # 停止语音相关服务
        self.loop.create_task(self.shutdown_services())
        self.loop.call_later(1, lambda: QApplication.quit())
        event.accept()
        
    async def shutdown_services(self):
        """关闭所有服务"""
        try:
            # 停止语音合成服务
            if hasattr(self, 'tts_streamer'):
                await self.tts_streamer.shutdown()
                
            # 关闭语音识别
            if hasattr(self, 'asr_helper'):
                self.asr_helper.close_audio()
                
            # 保存对话记录
            if hasattr(self, 'conversation_manager'):
                await self.conversation_manager.save_tracking_data()
                
            logging.info("所有服务已关闭")
        except Exception as e:
            logging.error(f"关闭服务时出错: {e}")


# ==================================
# 全局错误处理
# ==================================
def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    """全局异常日志记录"""
    logging.error("未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = log_uncaught_exceptions

# ==================================
# 主程序入口
# ==================================
async def main_async():
    """异步主函数"""
    # 初始化全局TTS和QA服务
    tts = TTSStreamer()
    await tts.start_speech_processor()
    
    # 初始化应用
    app = QApplication(sys.argv)
    window = SweetPotatoGUI(user_name="甘薯爱好者")
    
    # 退出清理
    try:
        exit_code = app.exec()
    finally:
        # 确保资源被清理
        await tts.shutdown()
        
    return exit_code

def main():
    """主函数入口点"""
    try:
        # 设置高DPI支持
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # 创建事件循环
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        # 运行异步主函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        exit_code = loop.run_until_complete(main_async())
        
        sys.exit(exit_code)
    except Exception as e:
        logging.error(f"主程序出错: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

