import logging
import time
import asyncio
import json
import random
import os
import re
from llama_cpp import Llama
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import nest_asyncio
from openai import AsyncOpenAI, OpenAI
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from conversation import ConversationManager

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat.log"), logging.StreamHandler()]
)

class LlamaCppEmbeddings(Embeddings):
    """自定义嵌入类，使用 llama.cpp 加载 GGUF 模型生成嵌入"""
    def __init__(self, model_path):
        
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
        conversation_manager=None,
        model_name="qwen-turbo-latest",
        api_key='',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        mcp_config_path="/home/wuye/vscode/raspberrypi_5/rasoberry/mcp_server_config.json"
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
        
        # MCP配置初始化
        self.mcp_config_path = mcp_config_path
        self.load_mcp_config()

        # 仅保留最基本的Agent配置, 使用与test.py相似的方式处理API调用
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
        ,"月","号","年","天","那天"
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
            # 将参数转换为JSON字符串
            tool_args_str = json.dumps(tool_args, ensure_ascii=False)
            result = self.bot._call_tool(tool_name, tool_args_str)
            return result
        except Exception as e:
            logging.error(f"调用工具 {tool_name} 失败: {e}")
            return {"error": f"调用工具失败: {str(e)}"}
    
    def detect_music_intent(self, question):
        """检测音乐相关意图和具体命令"""
        question_lower = question.lower()
        
        # 首先检查特定的音乐控制命令（优先级更高）
        specific_commands = {
            "播放列表": ["播放列表", "显示播放列表", "当前播放列表", "歌单", "列表", "有什么歌"],
            "下一首": ["下一首", "换一首", "下首歌", "播放下一首", "换一首歌", "播放另一首", 
                    "我想听其他的歌", "播放不同的歌", "换首歌", "不好听", "太难听了", 
                    "换个风格", "这歌不好听", "换别的", "换一个", "不喜欢这首"],
            "上一首": ["上一首", "前一首", "播放上一首", "返回上一首"],
            "暂停": ["暂停", "停下", "先停", "停一下"],
            "继续": ["继续", "恢复", "接着放", "再放", "继续播放"],
            "停止": ["停止播放", "关闭音乐", "不听了", "停止", "关掉", "退出播放"]
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
            (r"我想听\s*(.+)", "播放")
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
        """优化的搜索意图检测 - 提高优先级和准确性"""
        if not question:
            return None
            
        question_lower = question.lower().strip()
        
        # 明确的搜索命令 - 最高优先级
        explicit_search_commands = [
            "搜索", "查找", "查询", "搜一下", "查一下", "搜一搜",
            "帮我查", "帮我搜", "请搜索", "请查找", "搜索一下"
        ]
        
        # 检查是否以搜索命令开头
        for cmd in explicit_search_commands:
            if question_lower.startswith(cmd):
                # 提取搜索内容
                search_query = question_lower[len(cmd):].strip()
                # 清理可能的冗余词汇
                for kw in ["一下", "看看", "吧", "呢"]:
                    search_query = search_query.replace(kw, "").strip()
                    
                if search_query:
                    return {"command": "search", "query": search_query}
                return {"command": "search", "query": question}
        
        # 检查是否包含搜索命令（不仅是开头）
        for keyword in explicit_search_commands:
            if keyword in question_lower:
                # 更精确的提取逻辑
                if "帮我" in question_lower:
                    # 处理"帮我搜索XXX"格式
                    match = re.search(r"帮我.*(搜索|查找|查询)(.+)", question_lower)
                    if match:
                        search_query = match.group(2).strip()
                        return {"command": "search", "query": search_query}
                else:
                    # 处理"搜索XXX"格式
                    patterns = [
                        f"{keyword}\\s*(.+)",
                        f"请{keyword}\\s*(.+)"
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, question_lower)
                        if match:
                            search_query = match.group(1).strip()
                            # 清理查询词
                            for kw in explicit_search_commands:
                                search_query = search_query.replace(kw, "").strip()
                            
                            if search_query:
                                return {"command": "search", "query": search_query}
                
                return {"command": "search", "query": question}
        
        # 检测需要实时信息的关键词
        web_info_keywords = [
            "最新", "最近", "现在", "今天", "目前", "当前", "实时", 
            "新闻", "热点", "天气", "股价", "比分", "排行", "趋势", 
            "动态", "更新", "价格", "明天", "后天", "昨天"
        ]
        
        for keyword in web_info_keywords:
            if keyword in question_lower:
                return {"command": "search", "query": question}
        
        # 检测明确需要网络查询的模式
        web_query_patterns = [
            r"(.+)是什么公司",
            r"(.+)是什么品牌", 
            r"(.+)怎么样",
            r"(.+)的价格",
            r"(.+)多少钱",
            r"(.+)新闻",
            r"(.+)最新消息",
            r"(.+)股价",
            r"(.+)官网"
        ]
        
        for pattern in web_query_patterns:
            if re.search(pattern, question_lower):
                return {"command": "search", "query": question}
        
        return None
        
    async def handle_search_command_stream(self, intent):
        """流式处理搜索命令"""
        command = intent.get("command")
        query = intent.get("query", "")
        
        if command == "search" and query:
            yield "正在执行网络搜索任务..."
            await asyncio.sleep(0.3)
            # yield f"正在搜索关键词：{query}..."
            await asyncio.sleep(0.5)
            
            try:
                # 调用搜索工具
                tool_result = await self.call_tool("web_search-web_search", {"query": query, "limit": 5})
                
                # yield "搜索完成，正在处理结果..."
                await asyncio.sleep(0.2)
                
                # 解析搜索结果
                if isinstance(tool_result, str):
                    search_data = json.loads(tool_result)
                else:
                    search_data = tool_result
                
                # 检查搜索状态
                if search_data.get("status") == "error":
                    # yield f"搜索失败：{search_data.get('message', '未知错误')}"
                    return
                
                # 流式输出搜索结果
                if search_data.get("status") == "success" and "results" in search_data:
                    results = search_data["results"]
                    if results:
                        if len(results) == 1:
                            content = results[0].get("content", "搜索结果为空")
                            # 分块输出长文本
                            if len(content) > 50:
                                chunk_size = 40
                                for i in range(0, len(content), chunk_size):
                                    chunk = content[i:i + chunk_size]
                                    yield chunk
                                    await asyncio.sleep(0.08)  # 流式输出延迟
                            else:
                                yield content
                        else:
                            # 多个结果时流式输出
                            
                            await asyncio.sleep(0.02)
                            
                            for i, result in enumerate(results[:3], 1):
                                content = result.get("content", "")
                                if content:
                                    header = f"{i}. "
                                    yield header
                                    await asyncio.sleep(0.1)
                                    
                                    # 分块输出内容
                                    chunk_size = 35
                                    for j in range(0, len(content), chunk_size):
                                        chunk = content[j:j + chunk_size]
                                        yield chunk
                                        await asyncio.sleep(0.08)
                                    yield "\n\n"
                                    await asyncio.sleep(0.1)
                    else:
                        yield f"搜索({query})未找到相关结果。"
                else:
                    # 如果返回格式不符合预期，尝试直接返回
                    yield str(search_data)
                    
            except Exception as e:
                logging.error(f"处理搜索结果失败: {e}")
                yield f"搜索({query})时出现错误：{str(e)}"
        else:
            yield "请提供要搜索的内容。"
    
    async def handle_music_command(self, intent):
        """处理音乐相关命令"""
        command = intent.get("command")
        
        try:
            # 处理播放命令
            if command == "播放":
                song_name = intent.get("song_name", "")
                if not song_name:
                    return "请告诉我您想听的歌曲名称"
                    
                # 异步执行播放命令
                tool_result = await self.call_tool("netease_music-play_music", {"song_name": song_name})
                
                # 记录对话
                response_time = time.time() - self.current_question_start_time if hasattr(self, 'current_question_start_time') else 0
                if hasattr(self, 'conversation_manager'):
                    await self.conversation_manager.add_conversation_entry(
                        f"播放音乐: {song_name}", 
                        str(tool_result), 
                        response_time
                    )
                
                # 返回播放结果
                return f"正在为您播放{song_name}：{tool_result}"
                
            # 处理其他音乐控制命令
            elif command in ["暂停", "继续", "停止", "下一首", "上一首", "播放列表"]:
                # 为不同命令调用相应的工具
                tool_mapping = {
                    "暂停": ("netease_music-pauseplay", {}, "已暂停"),
                    "继续": ("netease_music-unpauseplay", {}, "已继续播放"),
                    "停止": ("netease_music-stopplay", {}, "已停止播放"),
                    "下一首": ("netease_music-next_song", {}, "已切换到下一首"),
                    "上一首": ("netease_music-previous_song", {}, "已切换到上一首"),
                    "播放列表": ("netease_music-get_playlist", {}, "当前播放列表")
                }
                
                tool_name, params, feedback = tool_mapping.get(command, (None, None, None))
                
                if tool_name:
                    # 异步执行命令
                    tool_result = await self.call_tool(tool_name, params)
                    
                    # 记录对话
                    if hasattr(self, 'conversation_manager') and hasattr(self, 'current_question_start_time'):
                        response_time = time.time() - self.current_question_start_time
                        await self.conversation_manager.add_conversation_entry(
                            f"音乐命令: {command}", 
                            str(tool_result), 
                            response_time
                        )
                    
                    # 返回执行结果
                    return f"{feedback}：{tool_result}"
            
            return "未能识别的音乐命令"
                
        except Exception as e:
            logging.error(f"处理音乐命令时出错: {e}")
            return f"处理音乐命令时出现错误: {str(e)}"

 
    
    def detect_camera_intent(self, question):
        """优化的摄像头意图检测 - 降低误触发率"""
        if not question:
            return None
            
        question_lower = question.lower().strip()
        
        # 1. 首先检查是否包含明确的排除关键词
        exclude_keywords = [
            # 农业相关
            "甘薯", "红薯", "地瓜", "种植", "栽培", "施肥", "病虫害",
            "品种", "产量", "营养", "土壤", "灌溉", "收获", "储存",
            # 搜索相关
            "搜索", "查找", "查询", "搜一下", "查一下", "搜一搜",
            "帮我查", "帮我搜", "请搜索", "请查找", "百度", "谷歌",
            # 知识问答
            "为什么", "怎么做", "如何", "什么原理", "解释", "说明",
            "历史", "起源", "发展", "区别", "对比", "分析",
            # 其他常见非视觉问题
            "价格", "多少钱", "哪里买", "推荐", "建议", "评价",
            "天气", "温度", "湿度", "时间", "日期", "新闻"
        ]
        
        # 如果包含排除关键词，直接返回None
        if any(keyword in question_lower for keyword in exclude_keywords):
            return None
        
        # 2. 明确的摄像头命令（高优先级）
        explicit_camera_commands = {
            "拍照识别": [
                 "看到","拍个照识别",  
                "看看", "拍照分析", "识别","这是","这里","手里","面前","镜头","相机","摄像头","眼前"
            ],
            "拍照": [
                "拍照", "拍张照", "拍个照", "照相", "拍一张", "来张照片",
                "给我拍照", "帮我拍照", "拍个图"
            ],
            "查看照片": ["查看照片", "看照片", "照片列表", "有哪些照片", "打开相册"],
            "摄像头状态": ["摄像头状态", "相机状态", "摄像头怎么样", "相机工作吗"]
        }
        
        # 检查明确命令
        for command, keywords in explicit_camera_commands.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return {"command": command, "original_question": question}
        
        # 3. 需要视觉识别的关键词组合（中等优先级）
        # 必须同时满足多个条件
        visual_context_words = ["面前", "手里", "手上", "桌上", "眼前", "镜头前", "这里", "那里"]
        visual_question_words = ["是什么", "什么东西", "什么玩意", "是啥"]
        
        # 检查是否有明确的视觉上下文
        has_visual_context = any(context in question_lower for context in visual_context_words)
        has_visual_question = any(q in question_lower for q in visual_question_words)
        
        # 必须同时包含视觉上下文和疑问词
        if has_visual_context and has_visual_question:
            return {"command": "拍照识别", "original_question": question}
        
        # 4. 明确需要视觉识别的完整短语（严格匹配）
        visual_recognition_patterns = [
            r"^.{0,5}(看看|瞧瞧|帮我看|帮我瞧).{0,5}(这|那|这个|这里|那个|我手里|面前|桌上).{0,5}(是什么|是啥|什么东西|什么牌子|什么品牌).*$",
            r"^.{0,5}(这个|那个|我手里的|桌上的|面前的|镜头|摄像机).{0,5}(东西|物品|物体).{0,5}(是什么|是啥).*$",
            r"^.{0,5}(帮我|请|能不能).{0,5}(看看|识别|认一下).{0,5}(这|那|这个|那个).*$",
            r"^.{0,5}(拍照).{0,5}(看看|识别|分析).{0,5}(这|那|什么).*$"
        ]
        
        for pattern in visual_recognition_patterns:
            if re.search(pattern, question_lower):
                return {"command": "拍照识别", "original_question": question}
        
        # 5. 数量识别的特定模式（严格的视觉场景）
        visual_quantity_patterns = [
            r"^.{0,10}(我|你).{0,5}(手指|手).{0,5}(比|举|伸).{0,5}(几个|多少个?).*$",
            r"^.{0,10}(数数|数一下|看看).{0,5}(我|面前|桌上|这里).{0,5}(有)?.{0,5}(几个|多少个?).*$",
            r"^.{0,10}(面前|桌上|手里|这里).{0,10}(有)?.{0,5}(几个|多少个?).*$",
            r"^.{0,10}(看看|瞧瞧).{0,5}(这|那|我).{0,5}(是|有).{0,5}(几个|多少).*$"
        ]
        
        for pattern in visual_quantity_patterns:
            if re.search(pattern, question_lower):
                # 额外检查：确保不是价格相关
                if not any(word in question_lower for word in ["多少钱", "价格", "成本", "费用", "售价"]):
                    return {"command": "拍照识别", "original_question": question}
        
       
        if len(question) < 15:  
            return None
        
        # 默认不触发摄像头
        return None
    

    def preprocess_camera_question(self, question):
        """预处理相机相关问题，提高意图识别准确性"""
        
        # 问题标准化映射
        question_mappings = {
            # 数量相关的口语化表达
            "这有几个": "这里有几个",
            "一共几个": "总共有几个",
            "多少个啊": "有多少个",
            "几个呀": "有几个",
            "数一数": "数数有多少个",
            
            # 位置相关的口语化表达
            "左边那个": "左边的是什么",
            "右边那个": "右边的是什么",
            "上边": "上面",
            "下边": "下面",
            
            # 动作相关的口语化表达
            "干啥呢": "在做什么",
            "干嘛呢": "在做什么",
            "搞什么": "在做什么",
            
            # 其他口语化表达
            "啥颜色": "什么颜色",
            "啥样子": "什么样子",
            "咋样": "怎么样",
            "是啥": "是什么",
            "有啥": "有什么"
        }
        
        # 应用映射进行标准化
        processed_question = question
        for oral, standard in question_mappings.items():
            if oral in processed_question:
                processed_question = processed_question.replace(oral, standard)
        
        return processed_question


    async def handle_camera_command_stream(self, intent):
        """流式处理摄像头命令"""
        command = intent.get("command")
        yield "正在拍摄照片..."
        original_question = intent.get("original_question", "")
        start_time = time.time()
        
        try:
            if command == "拍照":
                yield "正在启动摄像头..."
                await asyncio.sleep(0.1)
                result = await self._simple_photo()
                yield result
                
            elif command == "拍照识别": 
                # yield "正在启摄像头..."
                await asyncio.sleep(0.3)
                
                
                # yield "正在进行AI图像分析..."
                await asyncio.sleep(0.2)
                
                result = await self._photo_analysis(original_question)
                
                # 分块输出分析结果
                if result and len(result) > 50:
                    # 将长文本分块输出，每块约30-50字
                    words = result
                    chunk_size = 40
                    for i in range(0, len(words), chunk_size):
                        chunk = words[i:i + chunk_size]
                        yield chunk
                        await asyncio.sleep(0.1)  # 短暂延迟模拟流式效果
                else:
                    yield result
                    
            elif command == "查看照片":
                # yield "正在查询照片列表..."
                await asyncio.sleep(0.2)
                result = await self._list_photos()
                yield result
                
            elif command == "摄像头状态":
                # yield "正在检查摄像头状态..."
                await asyncio.sleep(0.2)
                result = await self._camera_status()
                yield result
            else:
                yield "摄像头命令无效"
            
            # 记录对话
            if hasattr(self, 'conversation_manager'):
                response_time = time.time() - start_time
                await self.conversation_manager.add_conversation_entry(
                    f"摄像头: {command}", result if 'result' in locals() else "操作完成", response_time
                )
            
        except Exception as e:
            logging.error(f"摄像头操作失败: {e}")
            yield "摄像头操作出错"


    async def _simple_photo(self):
        """简单拍照"""
        try:
            tool_result = await self.call_tool("camera-take_photo_only", {})
            result_data = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
            
            if result_data.get("status") == "success":
                return "拍照成功"
            else:
                return "拍照失败"
        except:
            return "拍照出错"

    async def _photo_analysis(self, user_question=""):
        """拍照分析 - 根据用户问题进行针对性分析，支持多种特殊场景"""
        try:
            # 定义各类问题的关键词和对应的提示词模板
            analysis_patterns = {
                # 数量计数类
                "quantity": {
                    "keywords": ["几个", "多少", "几根", "几只", "几条", "几张", "几块", "数数", "数一下", "有多少"],
                    "prompt_template": "请准确计数并直接回答：{question}。请给出具体数字。用聊天的方式简洁回答图片内容"
                },
                
                # 比较判断类
                "comparison": {
                    "keywords": ["哪个更", "谁更", "最大", "最小", "最高", "最矮", "最多", "最少", "比较"],
                    "prompt_template": "请比较图片中的对象并回答：{question}。请明确指出比较结果。用聊天的方式简洁回答图片内容"
                },
                
                # 位置方向类
                "location": {
                    "keywords": ["左边", "右边", "上面", "下面", "前面", "后面", "中间", "旁边", "位置", "哪里", "在哪"],
                    "prompt_template": "请根据图片中的位置关系回答：{question}。请明确说明方位。用聊天的方式简洁回答图片内容"
                },
                
                # 动作行为类
                "action": {
                    "keywords": ["在做什么", "什么动作", "正在", "怎么做", "在干嘛", "动作是"],
                    "prompt_template": "请描述图片中的动作或行为：{question}。请具体说明动作内容。用聊天的方式简洁回答图片内容"
                },
                
                # 情绪表情类
                "emotion": {
                    "keywords": ["什么表情", "开心吗", "难过吗", "生气", "高兴", "情绪", "心情", "感觉"],
                    "prompt_template": "请分析图片中的表情或情绪：{question}。请描述具体的情绪状态。用聊天的方式简洁回答图片内容"
                },
                
                # 文字识别类
                "text": {
                    "keywords": ["写的什么", "什么字", "文字内容", "上面写着", "标签", "文本", "标题","写了什么"],
                    "prompt_template": "请识别并读出图片中的文字内容：{question}。请准确转述所有可见文字。用聊天的方式简洁回答图片内容"
                },
                
                # 存在性判断类
                "existence": {
                    "keywords": ["有没有", "是否有", "存在", "能看到", "有无", "是不是有"],
                    "prompt_template": "请判断并回答：{question}。请明确回答'有'或'没有'，并说明具体情况。用聊天的方式简洁回答图片内容"
                },
                
                # 颜色外观类
                "appearance": {
                    "keywords": ["什么颜色", "颜色是", "什么样子", "长什么样", "外观", "形状"],
                    "prompt_template": "请描述外观特征来回答：{question}。请具体说明颜色、形状等特征。用聊天的方式简洁回答图片内容"
                },
                
                # 品牌标识类
                "brand": {
                    "keywords": ["什么牌子", "哪个品牌", "什么品牌", "商标", "logo", "标志"],
                    "prompt_template": "请识别品牌或标识：{question}。如果能识别出品牌，请明确说出品牌名称。用聊天的方式简洁回答图片内容"
                },
                
                # 时间相关类
                "time": {
                    "keywords": ["几点", "什么时间", "时间是", "显示时间", "钟表"],
                    "prompt_template": "请读取时间信息：{question}。如果图中有时间显示，请准确读出。用聊天的方式简洁回答图片内容"
                },
                
                # 相似度判断类
                "similarity": {
                    "keywords": ["像什么", "像不像", "是不是", "看起来像", "类似", "相似"],
                    "prompt_template": "请进行相似性判断：{question}。请说明相似或不相似的理由。用聊天的方式简洁回答图片内容"
                },
                
                # 材质属性类
                "material": {
                    "keywords": ["什么材质", "什么材料", "是金属", "是塑料", "是木头", "质地"],
                    "prompt_template": "请判断材质或质地：{question}。请根据视觉特征推断可能的材质。用聊天的方式简洁回答图片内容"
                },
                
                # 状态条件类
                "condition": {
                    "keywords": ["新的还是旧的", "完好", "破损", "干净", "脏", "整齐", "凌乱", "状态"],
                    "prompt_template": "请评估状态或条件：{question}。请描述具体的状态特征。用聊天的方式简洁回答图片内容"
                },
                
                # 功能用途类
                "function": {
                    "keywords": ["用来做什么", "什么用途", "干什么用的", "功能是", "用来"],
                    "prompt_template": "请说明功能或用途：{question}。请根据物品特征推断其可能的用途。用聊天的方式简洁回答图片内容"
                },
                
                # 安全相关类
                "safety": {
                    "keywords": ["危险吗", "安全吗", "有危险", "是否安全"],
                    "prompt_template": "请评估安全性：{question}。请指出可能的安全隐患或确认安全状态。用聊天的方式简洁回答图片内容"
                }
            }
            
            # 根据用户问题构建合适的提示词
            prompt = "用聊天的方式简洁回答图片内容"  # 默认提示词
            
            if user_question:
                question_lower = user_question.lower()
                
                # 遍历所有模式，找到匹配的类型
                matched = False
                for pattern_type, pattern_info in analysis_patterns.items():
                    keywords = pattern_info["keywords"]
                    if any(keyword in question_lower for keyword in keywords):
                        prompt = pattern_info["prompt_template"].format(question=user_question)
                        matched = True
                        logging.info(f"匹配到{pattern_type}类型的问题")
                        break
                
                # 如果没有匹配到特定模式，但有用户问题，使用通用问答模板
                if not matched and user_question:
                    prompt = f"请根据图片内容以聊天的方式回答以下问题：{user_question}"
            
            # 调用工具进行分析
            tool_result = await self.call_tool("camera-take_photo_and_analyze", {
                "prompt": prompt
            })
            
            # 处理返回结果
            if isinstance(tool_result, str):
                try:
                    result_data = json.loads(tool_result)
                    analysis = result_data.get("analysis", tool_result)
                except:
                    analysis = tool_result
            else:
                analysis = tool_result.get("analysis", "识别完成")
            
            # 使用改进的文本清理
            clean_text = self._clean_analysis_text(analysis)
            
            
            
            return clean_text
            
        except Exception as e:
            logging.error(f"拍照分析失败: {e}")
            return "识别失败"
            
        except Exception as e:
            logging.error(f"拍照分析失败: {e}")
            return "识别失败"
    async def _list_photos(self):
        """查看照片列表"""
        try:
            tool_result = await self.call_tool("camera-list_photos", {})
            result_data = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
            
            if result_data.get("status") == "success":
                total = result_data.get("total_photos", 0)
                if total == 0:
                    return "暂无照片"
                return f"共有{total}张照片"
            else:
                return "获取照片列表失败"
        except:
            return "查看照片出错"

    async def _camera_status(self):
        """摄像头状态"""
        try:
            tool_result = await self.call_tool("camera-get_camera_status", {})
            result_data = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
            
            if result_data.get("status") == "success":
                camera_status = result_data.get("camera_status", "未知")
                total_photos = result_data.get("total_photos", 0)
                return f"摄像头{camera_status}，已拍{total_photos}张照片"
            else:
                return "摄像头状态检查失败"
        except:
            return "状态检查出错"

    def _clean_analysis_text(self, text):
        """改进的文本清理 - 保留有用信息"""
        if not text or not isinstance(text, str):
            return "识别完成"
        
       
        
        # 打印原始返回结果用于调试
        # logging.info(f"原始分析结果: {text}")
        
        # 处理可能的JSON格式返回
        if text.startswith('{') and '"text"' in text:
            try:
             
                parsed_data = json.loads(text)
                if isinstance(parsed_data, dict) and 'text' in parsed_data:
                    text = parsed_data['text']
            except:
                pass
        
        # 基本清理：去除多余的格式
        text = text.strip()
        
        # 移除Markdown格式但保留内容
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*]\s*', '', text, flags=re.MULTILINE)
        
        # 清理多余空格和换行
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 如果文本为空或只有标点，返回默认值
        if not text or len(text.strip('。，')) < 2:
            return "识别完成。"
        
        # 确保以句号结尾
        if text and not text.endswith(('。', '！', '？')):
            text += '。'
        
        # logging.info(f"清理后结果: {text}")
        return text

    # ==================== 核心问答处理模块 ====================
        
    # 修正的流式回答方法 - 基于test.py的实现
    async def ask_stream(self, question, use_context=True, use_tools=True):
        """流式响应回答问题 - 支持摄像头和搜索流式输出"""
        start_time = time.time()
        
        try:
            # 工具意图检测与处理 - 使用流式版本
            if use_tools:
                # 音乐意图检测 - 保持原有逻辑
                music_intent = self.detect_music_intent(question)
                if music_intent:
                    result = await self.handle_music_command(music_intent)
                    yield result
                    
                    # 统一对话记录处理
                    response_time = time.time() - start_time
                    await self.conversation_manager.add_conversation_entry(question, result, response_time)
                    await self.conversation_manager.save_tracking_data()
                    return
                
                # 摄像头意图检测 - 使用流式版本
                camera_intent = self.detect_camera_intent(question)
                if camera_intent:
                    async for chunk in self.handle_camera_command_stream(camera_intent):
                        yield chunk
                    
                    # 对话记录在流式方法内部处理
                    await self.conversation_manager.save_tracking_data()
                    return
                
                # 搜索意图检测 - 使用流式版本
                search_intent = self.detect_search_intent(question)
                if search_intent:
                    async for chunk in self.handle_search_command_stream(search_intent):
                        yield chunk
                    
                    # 统一对话记录处理
                    response_time = time.time() - start_time
                    await self.conversation_manager.add_conversation_entry(question, "搜索完成", response_time)
                    await self.conversation_manager.save_tracking_data()
                    return
            
            # 知识库回答生成 - 保持原有流式逻辑
            context = ""
            if use_context:
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

            query = "你是一个甘薯专家，请你以说话的标准回答,请你根据参考内容回答，回答输出为一段，回答内容简洁，如果参考内容中没有相关信息，请回答'{}'。".format(random.choice(self.unknown_responses))
            
            doc_context = "\n\n".join([d.page_content for d in docs])
            
            if context:
                prompt = f"对话历史:\n{context}\n\n参考内容:\n{doc_context}\n\n当前问题:\n{question}\n\n要求:{query}\n\n"
            else:
                prompt = f"参考内容:\n{doc_context}\n\n问题:\n{question}\n\n要求:{query}\n\n"
            
            messages = [{"role": "user", "content": prompt}]
            
            try:
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
                
                # 对话记录
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

    # 修正的非流式回答方法 - 基于test.py的实现
    async def ask(self, question, use_context=True, use_tools=True):
        """非流式方式回答问题 - 基于test.py修改"""
        start_time = time.time()
        
        try:
            # 检测意图并处理特殊命令
            if use_tools:
                # 检测音乐命令
                music_intent = self.detect_music_intent(question)
                if music_intent:
                    return await self.handle_music_command(music_intent)
                
                # 检测摄像头命令
                camera_intent = self.detect_camera_intent(question)
                if camera_intent:
                    return await self.handle_camera_command(camera_intent)
                
                # 检测搜索命令
                search_intent = self.detect_search_intent(question)
                if search_intent:
                    return await self.handle_search_command(search_intent)
            
            # 非特殊指令处理 - 使用知识库回答
            context = ""
            if use_context:
                context = self.conversation_manager.get_conversation_context(max_context=3)
            
            # 获取相关文档 - 按照test.py的直接调用方式
            docs = self.vectorstore.as_retriever(search_kwargs={"k": self.k_documents}).invoke(question)
            
            if not docs:
                result = "未检索到相关内容。"
                response_time = time.time() - start_time
                await self.conversation_manager.add_conversation_entry(question, result, response_time)
                return result
                
            # 构建查询和上下文 - 与test.py保持一致
            query = "你是一个甘薯专家，请你以说话的标准回答，请你根据参考内容回答，回答输出为一段，回答内容简洁，如果参考内容中没有相关信息，请回答'{}'。".format(random.choice(self.unknown_responses))
            doc_context = "\n\n".join([d.page_content for d in docs])
           
            if context:
                prompt = f"对话历史:\n{context}\n\n参考内容:\n{doc_context}\n\n当前问题:\n{question}\n\n要求:{query}\n\n"
            else:
                prompt = f"参考内容:\n{doc_context}\n\n问题:\n{question}\n\n要求:{query}\n\n"

            # 使用Qwen API进行非流式调用 - 与test.py保持一致
            messages = [{"role": "user", "content": prompt}]
            
            try:
                # 确保使用与test.py相同的参数
                response = self.client.chat.completions.create(
                    model=self.model_name,  
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=400
                )
                
                result = response.choices[0].message.content
                response_time = time.time() - start_time
                await self.conversation_manager.add_conversation_entry(question, result, response_time)
                await self.conversation_manager.save_tracking_data()
                return result
                
            except Exception as e:
                logging.error(f"API调用失败: {e}")
                error_msg = f"API调用出错: {e}"
                await self.conversation_manager.record_error("API_ERROR", str(e))
                return error_msg
                
        except Exception as e:
            logging.error(f"处理问题失败: {e}")
            error_msg = f"处理问题时出错: {e}"
            await self.conversation_manager.record_error("PROCESS_ERROR", str(e))
            return error_msg
    
    def get_player_status(self):
        """获取音乐播放器状态"""
        try:
            return self.bot._call_tool("netease_music-isPlaying", "{}")
        except Exception as e:
            logging.error(f"获取播放器状态失败: {e}")
            return "not playing"
    
    def pause_play(self):
        """暂停/继续音乐播放"""
        try:
            status = self.get_player_status()
            if status == "playing":
                return self.bot._call_tool("netease_music-pauseplay", "{}")
            elif status == "not playing":
                return self.bot._call_tool("netease_music-unpauseplay", "{}")
            return "操作失败"
        except Exception as e:
            logging.error(f"暂停/继续播放失败: {e}")
            return "操作失败"

async def main():
    conversation_manager = ConversationManager(max_history=10)
    qa = KnowledgeQA(conversation_manager=conversation_manager)
    
    print("甘薯知识助手已启动，您可以开始提问（支持本地知识库、播放音乐、联网搜索、摄像头拍照）")
    print("输入'退出'或'exit'可以结束对话")
    
    while True:
        question = input("\n请输入问题：")
        
        if question.lower() in ['退出', 'exit', 'quit']:
            print("感谢使用，再见！")
            break
        
        print("思考中 ", end="", flush=True)
        
        async for part in qa.ask_stream(question):
            print(part, end="", flush=True)
        
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())