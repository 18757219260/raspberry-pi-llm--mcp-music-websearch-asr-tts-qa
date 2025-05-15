import logging
import time
import asyncio
import json
import random
import os
import re
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import nest_asyncio
from openai import AsyncOpenAI, OpenAI
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from conversation import ConversationManager

# Apply nest_asyncio to prevent runtime errors with asyncio
nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat.log"), logging.StreamHandler()]
)

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
        """处理音乐相关命令 - 修正版"""
        
        command = intent.get("command")
        
        if command == "播放":
            song_name = intent.get("song_name", "")
            if song_name:
                # 调用MCP播放音乐
                tool_result = await self.call_tool("netease_music-play_music", {"song_name": song_name})
                
                # 构建响应消息，但不阻塞等待TTS完成
                if isinstance(tool_result, dict) and "status" in tool_result:
                    return tool_result["status"] #+ " 您可以随时说暂停、下一首等控制播放。"
                    
                return f"正在为您播放{song_name}... "#您可以随时说暂停、下一首等控制播放。"
            else:
                return "请告诉我您想听的歌曲名称或歌手"
                
        elif command == "暂停":
            tool_result = await self.call_tool("netease_music-pauseplay", {})
            return str(tool_result)
            
        elif command == "继续":
            tool_result = await self.call_tool("netease_music-unpauseplay", {})
            return str(tool_result)
            
        elif command == "停止":
            tool_result = await self.call_tool("netease_music-stopplay", {})
            return str(tool_result)
            
        elif command == "下一首":
            tool_result = await self.call_tool("netease_music-next_song", {})
            return str(tool_result)
            
        elif command == "播放列表":
            tool_result = await self.call_tool("netease_music-get_playlist", {})
            return str(tool_result)
    
    # 修正的流式回答方法 - 基于test.py的实现
    async def ask_stream(self, question, use_context=True, use_tools=True):
        """使用流式响应回答问题 - 基于test.py修改"""
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
                    # 新增: 先返回搜索提示，让UI可以先显示或TTS播报
                    # yield "11执行网络搜索任务中"
                    
                    # 执行搜索
                    result = await self.handle_search_command(search_intent)
                    
                    # 返回搜索结果
                    yield result
                    
                    # 记录对话
                    response_time = time.time() - start_time
                    await self.conversation_manager.add_conversation_entry(question, result, response_time)
                    await self.conversation_manager.save_tracking_data()
                    return
            
            # 非特殊指令处理 - 使用知识库回答 (按照test.py中的方式实现)
            context = ""
            if use_context:
                context = self.conversation_manager.get_conversation_context(max_context=3)
            
            # 使用test.py中的方法获取文档
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

            # 构建查询提示（按照test.py的方式）
            query = "你是一个甘薯专家，请你以说话的标准回答，请你根据参考内容回答，回答输出为一段，回答内容简洁，如果参考内容中没有相关信息，请回答'{}'。".format(random.choice(self.unknown_responses))
            
            # 构建包含上下文的提示
            doc_context = "\n\n".join([d.page_content for d in docs])
            
            # 如果有对话历史，将其加入提示
            if context:
                prompt = f"对话历史:\n{context}\n\n参考内容:\n{doc_context}\n\n当前问题:\n{question}\n\n要求:{query}\n\n"
            else:
                prompt = f"参考内容:\n{doc_context}\n\n问题:\n{question}\n\n要求:{query}\n\n"
            
            # 使用Qwen API进行流式调用 - 与test.py保持一致
            messages = [
      
                {"role": "user", "content": prompt}
            ]
            
            try:
                # 确保使用与test.py相同的参数和格式
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
            messages = [
                self.sys_msg,
                {"role": "user", "content": prompt}
            ]
            
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
        

# if __name__ == "__main__":
#     # 测试代码
#     async def test_qa():
#         conversation_manager = ConversationManager(max_history=10)
#         knowledge_qa = KnowledgeQA(conversation_manager=conversation_manager)
        
#         questions = [
#             "湖州明天的天气",
#             "甘薯的别名",
            
#             "可以做什么？",
#             "吴家卓的高考成绩？",
#             "播放kanyewest的come to life",
#             "明天是否合适去拜佛",
            
#             "最近南浔古镇的客流量",
#             "最近湖州的趣事",
#             "暂停",
#             "继续播放",
#             # "停止播放"
#         ]
        
#         for question in questions:
#             print(f"\n用户: {question}")
#             print("助手: ", end="", flush=True)
            
#             async for part in knowledge_qa.ask_stream(question):
#                 print(part, end="", flush=True)
            
#             print("\n")  

#         summary = conversation_manager.get_session_summary()
#         print("\n对话总结：")
#         print(f"会话ID: {summary['session_id']}")
#         print(f"总时长: {summary['duration']}秒")
#         print(f"总问题数: {summary['total_questions']}")
#         print(f"平均响应时间: {summary['avg_response_time']}秒")
#         print(f"错误数: {summary['error_count']}")
    
#     # 运行测试
#     asyncio.run(test_qa())
async def main():

    conversation_manager = ConversationManager(max_history=10)

    qa = KnowledgeQA(conversation_manager=conversation_manager)
    
    print("甘薯知识助手已启动，您可以开始提问（支持本地知识库、播放音乐、联网搜索）")
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

