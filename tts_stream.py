import asyncio
import edge_tts
import subprocess
import io
import logging
import re
import time


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
    
    # Optimized wait_until_done Method


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
