import logging
import time
import asyncio
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from llama_cpp import Llama  
import nest_asyncio
import time
from openai import OpenAI
from dotenv import load_dotenv
import random
nest_asyncio.apply()



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat.log"), logging.StreamHandler()]
)

class LlamaCppEmbeddings(Embeddings):
    """自定义嵌入类，使用 llama.cpp 加载 GGUF 模型生成嵌入（保持不变）"""
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
        k_documents=1,
        embedding_model_path="/home/wuye/vscode/raspberrypi_5/text2vec_base_chinese_q8.gguf",
        env_file="./qwen.env"
    ):
        # 加载环境变量中的API密钥
        load_dotenv(env_file)
        
        self.faiss_index_path = faiss_index_path
        self.k_documents = k_documents
        self.temperature = temperature
        self.embedding_model = LlamaCppEmbeddings(model_path=embedding_model_path)
        self.vectorstore = self._load_vectorstore_with_retry()
        self.unknown_responses  = [
    "我不知道",
    "这个问题我无法回答",
    "抱歉我不太会",
    "我还不了解这方面。",
    "对不起，我没有这方面的资料。",
    "我不知道这个答案，不过你可以去问吴家卓",
    "好像不太会？",
    "我里个豆阿，你问出这么难的问题我怎么会呢？"
]
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
    
    async def ask_stream(self, question, context=""):
        docs = await asyncio.to_thread(
            self.vectorstore.as_retriever(search_kwargs={"k": self.k_documents}).invoke,
            question
        )

        if not docs:
            yield "未检索到相关内容。"
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
            
            for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content or ""
                    yield content
                    
        except Exception as e:
            logging.error(f"API调用失败: {e}")
            yield f"API调用出错: {e}"

    def ask(self, question):
        docs = self.vectorstore.as_retriever(search_kwargs={"k": self.k_documents}).invoke(question)
        if not docs:
            return "未检索到相关内容。"
        query="你是一个甘薯专家，请你以说话的标准回答，请你根据参考内容回答，回答输出为一段，回答内容简洁，如果参考内容中没有相关信息，请回答'{}'。".format(random.choice(self.unknown_responses))
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"这是参考内容:\n{context}\n\n这是问题:\n{question}\n这是要求{query}"

        # 使用Qwen API进行非流式调用
        messages = [
            self.sys_msg,
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-turbo",  # 根据需要选择合适的模型
                messages=messages,
                temperature=self.temperature,
                max_tokens=200
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"API调用失败: {e}")
            return f"API调用出错: {e}"


async def main():
    qa = KnowledgeQA()
    question = "甘薯的未来"

    # full = qa.ask(question)
    # print( full)

    async for part in qa.ask_stream(question):
        print(part, end="", flush=True)


        

if __name__ == "__main__":
    asyncio.run(main())