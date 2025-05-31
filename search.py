import os
import logging
from mcp.server.fastmcp import FastMCP
from http import HTTPStatus
from dashscope import Application
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 创建MCP服务器实例
mcp = FastMCP("web_search")

# 配置私网终端节点
os.environ['DASHSCOPE_HTTP_BASE_URL'] = ''

@mcp.tool()
def web_search(query: str, limit: int = 5) -> str:
    """
    网络搜索功能
    
    params: 
        query: 搜索查询词
        limit: 返回结果数量限制
    """
    try:
        # 调用阿里云通义千问搜索应用
        response = Application.call(
            api_key='',
            app_id='', 
            prompt=query
        )
        
        if response.status_code != HTTPStatus.OK:
            logger.error(f"搜索失败: {response.message}")
            return f"搜索失败: {response.message}"
        
        # 直接返回搜索结果文本
        result_text = response.output.text
        
        # 格式化搜索结果，保持与qa_model_easy的兼容性
        return json.dumps({
            "status": "success",
            "results": [
                {
                    "title": f"搜索结果：{query}",
                    "content": result_text,
                    "link": "",
                    "query": query
                }
            ]
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"搜索过程中出错: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": f"搜索出错: {str(e)}"
        }, ensure_ascii=False)

@mcp.tool()
def get_search_status() -> str:
    """
    获取搜索服务状态
    
    return: 
        服务状态信息
    """
    try:
        # 测试搜索功能是否正常
        test_response = Application.call(
            api_key='',
            app_id='',
            prompt='测试'
        )
        
        if test_response.status_code == HTTPStatus.OK:
            return "搜索服务运行正常"
        else:
            return f"搜索服务异常: {test_response.message}"
            
    except Exception as e:
        logger.error(f"检查搜索服务状态失败: {e}")
        return f"搜索服务状态检查失败: {str(e)}"

if __name__ == "__main__":
    logger.info("启动网络搜索MCP服务器...")
    print("搜索服务器正在运行")
    mcp.run(transport='stdio')