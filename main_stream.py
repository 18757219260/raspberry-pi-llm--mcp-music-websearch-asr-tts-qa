import sys
import os
import asyncio
import logging
import signal
import argparse
import time
import itertools
import threading
from qa_model_easy import KnowledgeQA
from asr import ASRhelper
from tts_stream import TTSStreamer  
from face_recognize import FaceRecognizer
import random   
from conversation import ConversationManager


# 配置日志 - 美化日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)8s │ %(message)s",
    handlers=[logging.FileHandler("chatbox.log"), logging.StreamHandler()]
)


class LoadingAnimation:
    def __init__(self, desc="加载中"):
        self.desc = desc
        self.done = False
        self.thread = None
        
    def animate(self):
        # 动画符号选项
        spinners = [
            "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
            "🌑🌒🌓🌔🌕🌖🌗🌘",
            ["[■□□□□□□]", "[■■□□□□□]", "[■■■□□□□]", "[■■■■□□□]","[■■■■■□□]", "[■■■■■■□]","[■■■■■■■]"],
            ["(•_•)", "( •_•)>⌐■-■", "(⌐■_■)"],
            ["🐱  ", " 🐱 ", "  🐱", " 🐱 "],
            ["🐶➡️", "🐶 ➡️", "🐶  ➡️", "🐶   ➡️"]
        ]
        spinner = spinners[2]  
        
        for char in itertools.cycle(spinner):
            if self.done:
                break
            sys.stdout.write(f"\r{char} {self.desc} ")
            sys.stdout.flush()
            time.sleep(0.3)
        # 清除加载动画行
        sys.stdout.write(f"\r{'✅ ' + self.desc + ' 完成!':60}\n")
        sys.stdout.flush()
        
    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.done = False
            self.thread = threading.Thread(target=self.animate)
            self.thread.start()
        
    def stop(self):
        self.done = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class AnimationManager:
    """动画管理器 - 确保同时只有一个动画在运行"""
    def __init__(self):
        self.current_animation = None
        self.lock = threading.Lock()
    
    def start_animation(self, desc):
        """启动新动画，自动停止之前的动画"""
        with self.lock:
            # 停止当前动画
            if self.current_animation:
                self.current_animation.stop()
            
            # 启动新动画
            self.current_animation = LoadingAnimation(desc)
            self.current_animation.start()
            return self.current_animation
    
    def stop_current(self):
        """停止当前动画"""
        with self.lock:
            if self.current_animation:
                self.current_animation.stop()
                self.current_animation = None


class SweetPotatoChatbox:
    def __init__(self, voice="zh-CN-XiaoyiNeural", debug=False):
        self.voice = voice
        self.debug = debug
        self.shutdown_event = asyncio.Event()
        self.qa = None
        self.tts = None
        self.asr = None
        self.face_auth_success = False
        self.recognized_user = None
        self.first_interaction = True 
        self.conversation_manager = ConversationManager() 
        self.current_question_start_time = None
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
        self.mcp_connected = False
        
        # 音乐交互状态管理
        self.music_interaction_mode = "normal"  # normal, waiting, real_time
        self.music_timer_task = None
        
        # 动画管理器
        self.animation_manager = AnimationManager()
        
    async def authenticate_user(self):
        """使用人脸识别进行用户认证"""
        logging.info("🔐 开始人脸认证...")
        print("\n📷 开始人脸认证，请面向摄像头...")
        
        # 初始化TTS用于提示信息
        temp_tts = TTSStreamer(voice=self.voice)
        await temp_tts.speak_text("11开始人脸认证，请面向摄像头", wait=True)
        time.sleep(0.5)
        
        await asyncio.sleep(1.0)
        
        # 初始化人脸识别系统
        face_system = FaceRecognizer()
        if not face_system.initialize():
            logging.error("❌ 人脸识别系统初始化失败")
            await temp_tts.speak_text("11人脸识别系统初始化失败,请检查人脸模型", wait=True)
            await temp_tts.shutdown()
            print("❌ 人脸识别系统初始化失败，程序退出")
            return False, None
        
        # 执行人脸认证
        auth_success, user_name = face_system.recognize_face()
        
        # 根据认证结果提供语音反馈
        if auth_success:
            self.conversation_manager.tracking_data['user_id'] = user_name
            welcome_message = f"11欢迎你{user_name}已进入甘薯知识系统。"
            logging.info(f"✅ 认证成功: {user_name}")
            print(f"\n✅ 认证成功！欢迎 {user_name}")
            await temp_tts.speak_text(welcome_message, wait=True)
        else:
            deny_message = "11你是谁我不认识你系统将退出。"
            logging.info("🚫 认证失败，拒绝访问")
            print("\n🚫 认证失败，无法识别用户，系统将退出")
            await temp_tts.speak_text(deny_message, wait=True)
        
        await temp_tts.shutdown()
        return auth_success, user_name
        
    async def initialize(self):
        """初始化所有组件"""
        try:
            logging.info("🚀 正在初始化甘薯问答系统...")
            print("\n🚀 正在初始化甘薯问答系统...")
            
            # 初始化TTS
            self.animation_manager.start_animation("初始化语音合成系统")
            self.tts = TTSStreamer(voice=self.voice)
            self.animation_manager.stop_current()
            
            try:
                await self.tts.speak_text("11正在初始化系统...", wait=True)
            except Exception as e:
                logging.error(f"⚠️ TTS初始化测试失败: {e}")
                print("⚠️ 警告: 语音合成服务不可用，将以文本方式提供反馈")
            
            await asyncio.sleep(0.5)
                
            # 初始化ASR
            logging.info("🎤 初始化语音识别...")
            self.animation_manager.start_animation("初始化语音识别系统")
            self.asr = ASRhelper()
            self.animation_manager.stop_current()
            
            # 初始化QA模型
            logging.info("🧠 正在加载知识模型，这可能需要一些时间...")
            try:
                await self.tts.speak_text("11正在加载知识模型，这可能需要一些时间...", wait=True)
            except Exception as e:
                logging.error(f"⚠️ TTS语音播放失败: {e}")
                print("🧠 正在加载知识模型，这可能需要一些时间...")
            
            await asyncio.sleep(0.5)
            
            # 显示加载动画
            self.animation_manager.start_animation("加载知识模型")
            self.qa = KnowledgeQA(conversation_manager=self.conversation_manager)
            self.animation_manager.stop_current()

            # MCP初始化
            self.animation_manager.start_animation("初始化MCP服务")
            self.mcp_connected = await self.tts.connect_to_mcp()
            self.animation_manager.stop_current()
            
            if self.mcp_connected:
                logging.info("✅ MCP服务已成功连接")
                print("✅ MCP服务已成功连接")
            else:
                logging.warning("⚠️ MCP服务连接失败，部分功能可能不可用")
                print("⚠️ MCP服务连接失败，部分功能可能不可用")
            
            logging.info("✨ 系统初始化完成")
            print("\n✨ 系统初始化完成，甘薯知识助手已准备就绪")
            return True
            
        except Exception as e:
            self.animation_manager.stop_current()
            logging.error(f"❌ 初始化失败: {e}")
            print(f"\n❌ 初始化失败: {e}")
            return False
            
    def setup_signal_handlers(self):
        """设置信号处理器用于优雅退出"""
        loop = asyncio.get_running_loop()
        for signame in ('SIGINT', 'SIGTERM'):
            loop.add_signal_handler(
                getattr(signal, signame),
                self.signal_handler
            )
    
    def signal_handler(self):
        """处理系统信号"""
        logging.info("🛑 收到系统退出信号，正在安全退出...")
        print(f"\n{'🛑 收到系统退出信号，正在安全退出... 🛑':^80}")
        self.shutdown_event.set()
    
    async def clear_audio_buffer(self):
        """清理音频缓冲区"""
        try:
            if hasattr(self.asr, 'stream'):
                time.sleep(0.1)
                while self.asr.stream.get_read_available() > 0:
                    self.asr.stream.read(self.asr.CHUNK, exception_on_overflow=False)
                logging.info("🧹 音频缓冲区已清理")
        except Exception as e:
            logging.warning(f"⚠️ 清理音频缓冲区时出错: {e}")

    async def get_music_preference(self,result):
        """询问用户对音乐播放的偏好设置"""
        logging.info("🎵 询问用户音乐播放偏好")
        
        # 询问用户偏好
        preference_prompt = f"{result}您希望等待播放完成再问问题，还是立即继续对话？"
        
        try:
            await self.tts.speak_text(preference_prompt, wait=True)
        except Exception as e:
            logging.error(f"⚠️ 播放偏好询问失败: {e}")
            print("🎵 音乐已开始播放，您希望等待播放完成再问问题，还是立即继续对话？")
        
        await asyncio.sleep(0.5)
        await self.clear_audio_buffer()
        
        # 显示监听指示器
        self.animation_manager.start_animation("正在聆听您的选择")
        
        # 获取用户回答
        preference_result = self.asr.real_time_recognition()
        self.animation_manager.stop_current()
        
        if not preference_result or 'result' not in preference_result or not preference_result['result']:
            logging.info("❌ 未检测到有效回答，默认选择立即继续")
            return "immediate"
        
        user_choice = preference_result["result"][0].lower()
        logging.info(f"🎵 用户音乐偏好选择: {user_choice}")
        
        # 解析用户选择
        if any(keyword in user_choice for keyword in ["等待", "等", "完成", "播放完","是的","没错","好","好的","继续","接着","听","放","歌"]):
            return "wait"
        elif any(keyword in user_choice for keyword in ["立即", "继续", "马上", "现在","提问","快","推进"]):
            return "immediate"
        elif any(keyword in user_choice for keyword in ["不确定", "不知道", "随便", "都行", "都可以","知道"]):
            return "uncertain"
        else:
            return "uncertain"

    async def handle_music_interaction(self, music_intent):
        """处理音乐相关的交互逻辑"""
        # 使用qa模型处理音乐指令
        result = await self.qa.handle_music_command(music_intent)
        
        logging.info(f"🎵 音乐操作结果: {result}")
        
        # 记录对话
        response_time = time.time() - self.current_question_start_time
        question = music_intent.get("song_name", "音乐操作")
        await self.conversation_manager.add_conversation_entry(question, result, response_time)
        await self.conversation_manager.save_tracking_data()
        
        # 如果是播放音乐命令，询问用户偏好
        if music_intent.get("command") == "播放":
            # 先播放音乐操作结果
            if result:
                clean_result = result.replace("11", "").strip()
                # await self.tts.speak_text(f"11{clean_result}", wait=True)
            
            # 询问用户偏好
            preference = await self.get_music_preference(result)
            
            if preference == "wait":
                self.music_interaction_mode = "waiting"
                await self.tts.speak_text("11将等待音乐播放完成后再继续。", wait=True)
                logging.info("🎵 设置模式: 等待音乐播放完成")
                
            elif preference == "immediate":
                self.music_interaction_mode = "real_time"
                await self.tts.speak_text("11好的，您可以随时发出语音指令。", wait=True)
                logging.info("🎵 设置模式: 实时交互")
                
            elif preference == "uncertain":
                # 创建一个专门用于定时器的新模式
                self.music_interaction_mode = "timer_waiting"  # <-- 修改此处
                await self.tts.speak_text("11好的，将在一分钟后询问您是否有问题。", wait=True)
                logging.info("🎵 设置模式: 定时提醒")
                
                # 启动定时器任务
                self.music_timer_task = asyncio.create_task(self.music_timer_reminder())
        else:
            # 非播放音乐命令，播放操作结果
            if result:
                clean_result = result.replace("11", "").strip()
                await self.tts.speak_text(f"11{clean_result}", wait=False)
        
        return True

    async def music_timer_reminder(self):
        try:
            await asyncio.sleep(60)
            
            # 定时器完成后不直接切换到normal模式，而是再次询问用户偏好
            if not self.shutdown_event.is_set() and self.music_interaction_mode == "timer_waiting":
                # 询问用户是否继续等待还是开始提问
                await self.tts.speak_text("11音乐正在播放，您希望等待播放完成再问问题，还是现在就开始提问？", wait=True)
                
                # 清理音频缓冲区
                await self.clear_audio_buffer()
                
                # 显示监听指示器
                self.animation_manager.start_animation("正在聆听您的选择")
                
                # 获取用户回答
                preference_result = self.asr.real_time_recognition()
                self.animation_manager.stop_current()
                
                if not preference_result or 'result' not in preference_result or not preference_result['result']:
                    logging.info("❌ 未检测到有效回答，继续等待")
                    # 如果没有有效回答，继续等待
                    self.music_timer_task = asyncio.create_task(self.music_timer_reminder())
                    return
                
                user_choice = preference_result["result"][0].lower()
                logging.info(f"🎵 用户音乐偏好选择: {user_choice}")
                
                # 解析用户选择
                if any(keyword in user_choice for keyword in ["等待", "等", "完成", "播放完", "是的", "没错", "好", "好的"]):
                    self.music_interaction_mode = "waiting"
                    await self.tts.speak_text("11好的，将等待音乐播放完成后再继续。", wait=True)
                    logging.info("🎵 设置模式: 等待音乐播放完成")
                elif any(keyword in user_choice for keyword in ["立即", "继续", "马上", "现在", "提问", "快", "推进"]):
                    self.music_interaction_mode = "real_time"
                    await self.tts.speak_text("11好的，您可以随时发出语音指令。", wait=True)
                    logging.info("🎵 设置模式: 实时交互")
                elif any(keyword in user_choice for keyword in ["不确定", "不知道", "随便", "都行", "都可以"]):
                    # 继续使用timer_waiting模式并重启定时器
                    self.music_timer_task = asyncio.create_task(self.music_timer_reminder())
                    await self.tts.speak_text("11好的，将在一分钟后再次询问。", wait=True)
                    logging.info("🎵 设置模式: 继续定时提醒")
                else:
                    # 默认保持当前模式并重启定时器
                    self.music_timer_task = asyncio.create_task(self.music_timer_reminder())
                    await self.tts.speak_text("11好的，将在一分钟后再次询问。", wait=True)
                    logging.info("🎵 设置模式: 继续定时提醒")
        except asyncio.CancelledError:
            logging.info("🎵 定时提醒任务被取消")
        except Exception as e:
            logging.error(f"🎵 定时提醒任务出错: {e}")

    async def process_user_input(self):
        """处理用户语音输入"""
        logging.info("\n🎤 等待语音播放完🎤")
        
        # 确保TTS完全结束
        await self.tts.wait_until_done()
        
        # 清空音频缓冲区
        await self.clear_audio_buffer()
        
        # 根据音乐交互模式决定是否询问
        if self.music_interaction_mode == "normal" or self.music_interaction_mode == "real_time":
            # 提示文本
            prompt_text = "11请问您有什么关于甘薯的问题？" if self.first_interaction else "11" + random.choice(self.follow_up_prompts)
            self.first_interaction = False
            
            try:
                await self.tts.speak_text(prompt_text, wait=True)
            except Exception as e:
                logging.error(f"⚠️ 语音提示失败: {e}")
                print(prompt_text.replace("11", ""))
            
            await asyncio.sleep(0.3)
            await self.clear_audio_buffer()


        elif self.music_interaction_mode == "timer_waiting":
            # 简单等待并返回None以循环而不提示
            await asyncio.sleep(2)
            return None
        
        elif self.music_interaction_mode == "waiting":
            # 等待模式：检查音乐是否还在播放
            player_status = self.qa.get_player_status()
            if player_status == "playing":
                # 音乐还在播放，继续等待
                logging.info("🎵 音乐正在播放，继续等待...")
                await asyncio.sleep(2)
                return None
            else:
                # 音乐播放完成，切换到正常模式
                self.music_interaction_mode = "normal"
                await self.tts.speak_text("11音乐播放完成，现在可以提问了。", wait=True)
                await self.clear_audio_buffer()
        
        # 显示监听指示器
        self.animation_manager.start_animation("正在聆听")
        
        # 执行语音识别
        question_result = self.asr.real_time_recognition()
        
        # 停止监听指示器
        self.animation_manager.stop_current()
        
        # 检查语音识别结果
        if  question_result=="" or 'result' not in question_result or not question_result['result'] or question_result=="嗯嗯" or question_result=="嗯嗯嗯嗯" or question_result=="嗯嗯嗯":
            logging.info("❌ 未检测到有效语音输入")
            print("❌ 未检测到有效语音输入")

            # 不进行TTS提示，直接等待10秒后继续
            logging.info("🕙 等待10秒后继续监听...")
            await asyncio.sleep(10)
            return None
            
        question = question_result["result"][0] 
        logging.info(f"💬 问题: {question}")
        self.current_question_start_time = time.time()


    
        
        if any(word in question.lower() for word in ["没", "无", "关闭", "拜拜", "再见", "退出"]):
            logging.info("="*80)
            logging.info(f"🚪 收到退出命令: '{question}'")
            logging.info("="*80)
            
            # 取消定时器任务
            if self.music_timer_task and not self.music_timer_task.done():
                self.music_timer_task.cancel()
            
            try:
                await self.tts.speak_text("11好的，感谢使用甘薯知识助手，再见！", wait=True)
            except:
                print("👋 感谢使用甘薯知识助手，再见！")
                
            self.shutdown_event.set()
            return None

        # 处理音乐相关指令
        if question:
            music_intent = self.qa.detect_music_intent(question)
            if music_intent:
                # 取消之前的定时器任务
                if self.music_timer_task and not self.music_timer_task.done():
                    self.music_timer_task.cancel()
                
                # 使用统一的音乐处理动画
                self.animation_manager.start_animation("正在处理音乐请求")
                
                try:
                    await self.handle_music_interaction(music_intent)
                finally:
                    self.animation_manager.stop_current()
                
                return None

        # 如果不是音乐命令，继续处理为普通问答
        return question
            
    async def run(self):
        """运行主循环"""
        # 首先进行人脸认证
        self.face_auth_success, self.recognized_user = await self.authenticate_user()
        
        # 如果人脸认证失败，退出程序
        if not self.face_auth_success:
            return
            
        # 人脸认证成功后，初始化系统组件
        if not await self.initialize():
            return
            
        self.setup_signal_handlers()
        
        # 启动提示
        print("\n" + "═" * 80)
        print(f"{'🌟 甘薯知识问答系统已启动 🌟':^80}")
        print(f"{'👤 用户: ' + self.recognized_user:^80}")
        print(f"{'⌨️  按 Ctrl+C 退出':^80}")
        print("═" * 80 + "\n")
        
        try:
            # 初始欢迎语
            try:
                await self.tts.speak_text(f"11{self.recognized_user}，甘薯知识问答系统已启动。", wait=True)
                await asyncio.sleep(0.5)
                await self.clear_audio_buffer()
            except Exception as e:
                logging.error(f"⚠️ 播放欢迎消息失败: {e}")
                print(f"👋 {self.recognized_user}，甘薯知识问答系统已启动。")

            while not self.shutdown_event.is_set():
                # 获取用户问题
                question = await self.process_user_input()
                
                # 检查是否收到退出信号
                if self.shutdown_event.is_set():
                    break
                
                # 处理问题并回答
                if question:
                    try:
                        # 统一的思考动画
                        self.animation_manager.start_animation("正在思考")
        
                        first_chunk = True
                        full_answer = ""
                        search_animation_started = False
                        
                        # 将上下文传递给 ask_stream 方法
                        async for chunk in self.qa.ask_stream(question):
                            if first_chunk and chunk.startswith("11正在执行网络搜索任务"):
                                await self.tts.speak_text("正在开启网络搜索任务",wait=False)
                                # 切换到搜索动画，只切换一次
                                if not search_animation_started:
                                    self.animation_manager.start_animation("执行网络搜索")
                                    search_animation_started = True
                                
                               
                                first_chunk = False
                                continue
                            else:
                                if first_chunk:
                                    first_chunk = False
                                full_answer += chunk
                        
                        # 停止加载动画
                        self.animation_manager.stop_current()
                        
                        response_time = time.time() - self.current_question_start_time
                        
                        # 异步记录对话
                        asyncio.create_task(
                            self.conversation_manager.add_conversation_entry(
                                question, full_answer, response_time
                            )
                        )
            
                        # 播放答案
                        if full_answer:
                            logging.info(f"💡 答案: {full_answer}")
                            
                            try:
                                await self.tts.speak_text(full_answer, wait=False)
                            except Exception as e:
                                logging.error(f"⚠️ 播放答案失败: {e}")
                                print(f"⚠️ 播放答案失败: {e}")
                                
                    except Exception as e:
                        self.animation_manager.stop_current()
                        logging.error(f"❌ 处理问题时出错: {e}")
                        print(f"❌ 处理问题时出错: {e}")
                    
        except KeyboardInterrupt:
            logging.info("⌨️ 收到键盘中断信号")
            print("\n⌨️ 收到键盘中断信号")
            self.shutdown_event.set()
        except asyncio.CancelledError:
            logging.info("🛑 任务被取消")
            print("\n🛑 任务被取消")
            self.shutdown_event.set()
        except Exception as e:
            logging.error(f"❌ 运行时发生错误: {e}")
            print(f"\n❌ 运行时发生错误: {e}")
        finally:
            # 取消定时器任务
            if self.music_timer_task and not self.music_timer_task.done():
                self.music_timer_task.cancel()
            # 停止所有动画
            self.animation_manager.stop_current()
            # 清理资源
            await self.shutdown()
            
    async def shutdown(self):
        """清理资源并关闭系统"""
        logging.info("🧹 正在关闭系统...")
        print("\n🧹 正在关闭系统...")
        
        # 显示关闭动画
        self.animation_manager.start_animation("正在清理系统资源")
        
        try:
            await self.conversation_manager.save_tracking_data()
            session_summary = self.conversation_manager.get_session_summary()
            logging.info(f"📊 会话统计: {session_summary}")
            print(f"\n📊 会话摘要:")
            print(f"  - 会话ID: {session_summary['session_id']}")
            print(f"  - 总时长: {session_summary['duration']}秒")
            print(f"  - 问题数: {session_summary['total_questions']}")
            print(f"  - 平均响应时间: {session_summary['avg_response_time']}秒")
            print(f"  - 错误次数: {session_summary['error_count']}")

            if self.tts and not self.shutdown_event.is_set():
                try:
                    await self.tts.speak_text("11感谢使用甘薯知识助手再见！", wait=True)
                except Exception as e:
                    logging.error(f"⚠️ 播放告别语音失败: {e}")
            
            # 先关闭TTS
            if self.tts:
                await self.tts.shutdown()
                
            # 关闭ASR
            if self.asr:
                self.asr.stop_recording()
                logging.info("✅ ASR资源已释放")
                
            # 停止关闭动画
            self.animation_manager.stop_current()
                
            logging.info("✅ 所有资源已清理，系统已安全关闭")
            print("\n" + "═" * 80)
            print(f"{'👋 系统已安全关闭，感谢使用！👋':^80}")
            print("═" * 80)
            
        except Exception as e:
            # 确保动画停止
            self.animation_manager.stop_current()
            logging.error(f"❌ 清理资源时出错: {e}")
            print(f"\n❌ 清理资源时出错: {e}")


async def main():
    """程序入口点"""
    # 显示启动横幅
    print("\n" + "═" * 80)
    print(f"{'🚀 甘薯知识问答系统 v2.0 🚀':^80}")
    print(f"{'启动中...':^80}")
    print("═" * 80 + "\n")
    
    parser = argparse.ArgumentParser(description="甘薯知识问答系统")
    parser.add_argument("--voice", default="zh-CN-XiaoyiNeural", help="TTS语音")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()
    
    try:
        chatbox = SweetPotatoChatbox(
            voice=args.voice,
            debug=args.debug
        )
        await chatbox.run()
    except KeyboardInterrupt:
        print("\n⌨️ 程序被用户中断")
    except Exception as e:
        logging.error(f"❌ 程序出错: {e}")
        print(f"\n❌ 程序出错: {e}")
    finally:
        print("\n👋 程序已完全退出")
        os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())