import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mk_faiss.log"), logging.StreamHandler()]
)

class MkFaiss:
    """向量存储管理器，用于知识库嵌入创建和管理"""
    
    def __init__(
        self,
        knowledge_path: str = "/home/wuye/vscode/raspberrypi_5/rasoberry/knowledge.json",
        faiss_index_path: str = "faiss_index",
        embedding_model_path: str = "/home/wuye/vscode/raspberrypi_5/rasoberry/text2vec_base_chinese_q8.gguf",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        初始化向量存储管理器
        
        参数:
            knowledge_path: 知识库JSON文件路径
            faiss_index_path: FAISS索引存储/加载路径
            embedding_model_path: text2vec-base-chinese-q8.gguf模型路径
            chunk_size: 文档分块大小
            chunk_overlap: 连续分块之间的重叠部分
        """
        self.knowledge_path = knowledge_path
        self.faiss_index_path = faiss_index_path
        self.embedding_model_path = embedding_model_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        

        
        try:
            self.embedding_model = self._init_embeddings()
            # 修改为直接创建向量存储，不再尝试加载现有的
            self.vectorstore = None
        except Exception as e:
            logging.error(f"向量存储初始化失败: {e}")
            raise
            
    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            if not os.path.exists(self.embedding_model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.embedding_model_path}")
            
            return LlamaCppEmbeddings(
                model_path=self.embedding_model_path,
                # 根据实际需求调整以下参数
                n_ctx=512,  # 上下文长度
                n_gpu_layers=0,  # 使用GPU加速的层数，0表示只使用CPU
                use_mlock=True,  # 防止模型被换出内存
                verbose=False  # 减少输出噪音
            )
        except Exception as e:
            logging.error(f"加载嵌入模型失败 {self.embedding_model_path}: {e}")
            raise RuntimeError(f"嵌入模型加载失败: {e}")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """从JSON文件加载知识库"""
        try:
            if not os.path.exists(self.knowledge_path):
                logging.error(f"知识库文件 {self.knowledge_path} 不存在")
                raise FileNotFoundError(f"知识库文件 {self.knowledge_path} 不存在")
                
            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logging.error("知识库必须是JSON数组格式")
                raise ValueError("知识库必须是JSON数组格式")
                
            if not data:
                logging.warning("知识库为空")
                
            logging.info(f"成功加载知识库: {len(data)} 条记录")
            return data
            
        except json.JSONDecodeError as e:
            logging.error(f"解析 {self.knowledge_path} 失败: {e}")
            raise
        except Exception as e:
            logging.error(f"加载知识库失败: {e}")
            raise
    
    def create_documents(self, knowledge_base: List[Dict[str, Any]]) -> List[Document]:
        """将知识库条目转换为Document对象"""
        docs = []
        for item in knowledge_base:
            question = item.get("question", "")
            answers = item.get("answer", [])
            
            if not question or not answers:
                logging.warning(f"无效的问答对: question={question}, answers={answers}")
                continue
                
            for answer in answers:
                if not isinstance(answer, str):
                    logging.warning(f"答案必须是字符串: {answer}")
                    continue
                    
                docs.append(Document(
                    page_content=f"Q: {question}\nA: {answer}",
                    metadata={
                        "question": question,
                        "source": self.knowledge_path,
                        "create_time": datetime.now().isoformat()
                    }
                ))
                
        logging.info(f"生成了 {len(docs)} 个文档")
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割为向量存储的块"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True
        )
        chunks = splitter.split_documents(documents)
        logging.info(f"生成了 {len(chunks)} 个文档分块")
        return chunks
    
    def load_or_create_vectorstore(self):
        """加载现有向量存储或创建新的向量存储"""
        try:
            # 直接创建新的向量数据库，而不是尝试加载现有的
            logging.info("创建新的向量数据库")
            data = self.load_data()
            docs = self.create_documents(data)
            
            if not docs:
                logging.error("无法创建向量数据库：文档为空")
                raise ValueError("文档为空")
                
            chunks = self.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            vectorstore.save_local(self.faiss_index_path)
            logging.info(f"向量数据库已保存到 {self.faiss_index_path}")
            return vectorstore
            
        except Exception as e:
            logging.error(f"创建向量数据库失败: {e}")
            raise RuntimeError(f"向量数据库处理失败: {e}")
    
    def update_knowledge(self) -> bool:
        """从知识库更新向量存储"""
        try:
            logging.info("开始更新知识库")
            data = self.load_data()
            docs = self.create_documents(data)
            
            if not docs:
                logging.error("无法更新向量数据库：文档为空")
                raise ValueError("文档为空")
                
            chunks = self.split_documents(docs)
            self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            self.vectorstore.save_local(self.faiss_index_path)
            logging.info("知识库更新成功！")
            return True
            
        except Exception as e:
            logging.error(f"更新知识库失败: {e}")
            return False
    
    def check_and_update_if_needed(self) -> bool:
        """直接更新知识库，不再检查时间戳"""
        if not os.path.exists(self.knowledge_path):
            logging.warning(f"知识库文件 {self.knowledge_path} 不存在")
            return False
            
        logging.info("直接更新知识库...")
        return self.update_knowledge()
    
    def get_vectorstore(self):
        """获取当前向量存储"""
        return self.vectorstore


if __name__ == "__main__":
    """命令行运行示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description="知识库向量构建工具")
    parser.add_argument("--knowledge", default="/home/wuye/vscode/raspberrypi_5/rasoberry/knowledge.json", help="知识库JSON文件路径")
    parser.add_argument("--index", default="faiss_index", help="FAISS索引目录路径")
    parser.add_argument("--model", default="/home/wuye/vscode/raspberrypi_5/rasoberry/text2vec_base_chinese_q8.gguf", help="text2vec-base-chinese-q8.gguf模型路径")
    parser.add_argument("--force", action="store_true", help="此参数已不再使用，保留仅为兼容性")
    args = parser.parse_args()
    
    try:
        mk_faiss = MkFaiss(
            knowledge_path=args.knowledge,
            faiss_index_path=args.index,
            embedding_model_path=args.model
        )
        
        # 直接更新知识库，无需条件判断
        logging.info("更新知识库...")
        success = mk_faiss.update_knowledge()
        if success:
            logging.info("知识库更新完成")
        else:
            logging.error("知识库更新失败")
                
        vectorstore = mk_faiss.get_vectorstore()
        logging.info("向量数据库测试成功")
        
    except Exception as e:
        logging.error(f"程序执行出错: {e}")