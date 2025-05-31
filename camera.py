import cv2
import base64
import logging
import os
import time
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from dashscope import MultiModalConversation
import json
from pathlib import Path

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 创建MCP服务器实例
mcp = FastMCP("camera")

# 全局配置
PHOTOS_DIR = Path("photos")
PHOTOS_DIR.mkdir(exist_ok=True)

class CameraManager:
    """摄像头管理器 - 负责摄像头初始化、拍照和资源管理"""
    
    def __init__(self, max_photos=10):
        self.camera = None
        self.is_initialized = False
        self.max_photos = max_photos
        
    def clear_camera_buffer(self):
        """清空摄像头缓冲区，确保获取最新帧"""
        if not self.camera or not self.camera.isOpened():
            return
            
        # 读取并丢弃多个帧以清空缓冲区
        # 这个数量可能需要根据实际情况调整
        for _ in range(5):
            self.camera.grab()  # grab() 比 read() 更快，只抓取不解码
            
        # 短暂延迟确保新帧到达
        time.sleep(0.1)
        
        logger.info("已清空摄像头缓冲区")

    def initialize_camera(self) -> bool:
        """初始化摄像头设备"""
        try:
            # 尝试多个摄像头索引
            for camera_index in range(3):
                self.camera = cv2.VideoCapture(camera_index)
                if self.camera.isOpened():
                    # 设置摄像头参数优化
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.camera.set(cv2.CAP_PROP_FPS, 30)
                    
                    # 预热摄像头
                    for _ in range(5):
                        ret, _ = self.camera.read()
                        if ret:
                            break
                        time.sleep(0.1)
                    
                    self.is_initialized = True
                    logger.info(f"摄像头初始化成功，使用设备索引: {camera_index}")
                    return True
                else:
                    self.camera.release()

            logger.error("无法找到可用的摄像头设备")
            return False
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False
        
    def manage_photo_storage(self):
        """管理照片存储，确保不超过最大数量"""
        try:
            # 获取所有照片文件
            photos = list(PHOTOS_DIR.glob("photo_*.jpg"))
            
            # 如果照片数量达到或超过限制
            if len(photos) >= self.max_photos:
                # 按修改时间排序（最旧的在前）
                photos.sort(key=lambda x: x.stat().st_mtime)
                
                # 计算需要删除的照片数量
                photos_to_delete = len(photos) - self.max_photos + 1
                
                # 删除最旧的照片
                for i in range(photos_to_delete):
                    try:
                        photos[i].unlink()
                        logger.info(f"已删除旧照片: {photos[i].name}")
                    except Exception as e:
                        logger.error(f"删除照片失败 {photos[i].name}: {e}")
                        
        except Exception as e:
            logger.error(f"管理照片存储时出错: {e}")
    
    def capture_photo(self) -> tuple[bool, str, str]:
        """
        拍摄照片并保存
        
        Returns:
            tuple: (成功标志, 图片路径, 错误信息)
        """
        if not self.is_initialized:
            if not self.initialize_camera():
                return False, "", "摄像头初始化失败"
        
        try:
            # 在拍照前管理存储空间
            self.manage_photo_storage()
            
            # 清空摄像头缓冲区 - 关键修改
            self.clear_camera_buffer()
            
            # 现在读取最新的帧
            ret, frame = self.camera.read()
            if not ret:
                return False, "", "无法捕获图像"
            
            # 生成唯一文件名 - 添加毫秒以避免同名文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"photo_{timestamp}.jpg"
            photo_path = PHOTOS_DIR / filename
            
            # 确保文件路径是绝对路径
            photo_path = photo_path.absolute()
            
            # 保存图片，使用高质量压缩
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            success = cv2.imwrite(str(photo_path), frame, encode_params)
            
            if success:
                # 验证文件已写入
                if os.path.exists(photo_path) and os.path.getsize(photo_path) > 0:
                    logger.info(f"照片保存成功: {photo_path}")
                    
                    # 再次检查照片数量
                    current_photos = len(list(PHOTOS_DIR.glob("photo_*.jpg")))
                    logger.info(f"当前照片数量: {current_photos}/{self.max_photos}")
                    
                    return True, str(photo_path), ""
                else:
                    logger.error(f"照片文件验证失败: {photo_path}")
                    return False, "", "照片保存验证失败"
            else:
                return False, "", "图片保存失败"
                
        except Exception as e:
            logger.error(f"拍照过程出错: {e}")
            return False, "", f"拍照失败: {str(e)}"
    
    def release_camera(self):
        """释放摄像头资源"""
        if self.camera is not None:
            self.camera.release()
            self.is_initialized = False
            logger.info("摄像头资源已释放")


camera_manager = CameraManager(max_photos=10)

def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64格式"""
    try:
        # 添加日志以跟踪正在编码的文件
        # logger.info(f"正在编码图片: {image_path}")
        
        # 验证文件路径
        if not os.path.exists(image_path):
            logger.error(f"图片文件不存在: {image_path}")
            return ""
        
        # 获取文件信息
        file_stat = os.stat(image_path)
        # logger.info(f"图片文件信息 - 大小: {file_stat.st_size} bytes, 修改时间: {datetime.fromtimestamp(file_stat.st_mtime)}")
        
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # logger.info(f"图片编码成功: {image_path}, 编码长度: {len(encoded_string)}")
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        logger.error(f"图片编码失败 {image_path}: {e}")
        return ""

def analyze_image_with_qwen(image_path: str, prompt: str = None) -> str:
    """使用通义千问视觉模型分析图片"""
    try:
        # 简化默认提示词
        if prompt is None:
            prompt = "用聊天的方式根据问题来简洁描述图片内容"
        
        # 编码图片
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return "图片编码失败"
        
        # 构建消息内容
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {"image": base64_image}
                ]
            }
        ]
        
        # 调用通义千问视觉模型
        response = MultiModalConversation.call(
            model='qwen-vl-max',
            messages=messages,
            api_key=''
        )
        
        if response.status_code == 200:
            result = response.output.choices[0].message.content
            # logger.info("图片分析完成")
            return result
        else:
            logger.error(f"模型调用失败: {response.message}")
            return "图片分析失败"
            
    except Exception as e:
        logger.error(f"图片分析过程出错: {e}")
        return "图片分析出错"

@mcp.tool()
def take_photo_and_analyze(prompt = None) -> str:
    """拍照并进行AI分析"""
    try:
        # 拍摄照片
        success, photo_path, error_msg = camera_manager.capture_photo()
        
        if not success:
            return f"拍照失败: {error_msg}"
        
        # 添加短暂延迟，确保文件完全写入磁盘
        time.sleep(0.5)
        
        # 验证文件是否存在且可读
        if not os.path.exists(photo_path):
            logger.error(f"照片文件不存在: {photo_path}")
            return "照片保存失败"
        
        # 再次验证文件大小，确保写入完成
        file_size = os.path.getsize(photo_path)
        if file_size == 0:
            logger.error(f"照片文件为空: {photo_path}")
            return "照片保存不完整"
        
        # logger.info(f"准备分析照片: {photo_path}, 文件大小: {file_size} bytes")
        
        # 使用简化的提示词
        if prompt is None:
            prompt = "用聊天的方式根据问题来简洁描述图片内容"
        
        # AI分析图片
        analysis_result = analyze_image_with_qwen(photo_path, prompt)
        
        # 记录分析完成
        # logger.info(f"已分析照片: {photo_path}")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"拍照分析失败: {e}")
        return "拍照分析失败"

@mcp.tool()
def take_photo_only() -> str:
    """仅拍照不分析"""
    try:
        success, photo_path, error_msg = camera_manager.capture_photo()
        
        if success:
            return json.dumps({
                "status": "success",
                "photo_path": photo_path,
                "message": "照片拍摄成功",
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "error",
                "message": error_msg
            }, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"拍照失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"拍照失败: {str(e)}"
        }, ensure_ascii=False)

@mcp.tool()
def analyze_existing_photo(photo_path: str, prompt: str = None) -> str:
    """分析已有的照片"""
    try:
        if not os.path.exists(photo_path):
            return "指定的照片文件不存在"
        
        if prompt is None:
            prompt = "用聊天的方式根据问题来简洁描述图片内容"
        
        analysis_result = analyze_image_with_qwen(photo_path, prompt)
        return analysis_result
        
    except Exception as e:
        logger.error(f"照片分析失败: {e}")
        return "照片分析失败"

@mcp.tool()
def get_camera_status() -> str:
    """获取摄像头状态"""
    try:
        if camera_manager.is_initialized:
            status = "已就绪"
        else:
            # 尝试初始化以检测摄像头可用性
            if camera_manager.initialize_camera():
                status = "已就绪"
            else:
                status = "不可用"
        
        return json.dumps({
            "status": "success",
            "camera_status": status,
            "photos_directory": str(PHOTOS_DIR),
            "total_photos": len(list(PHOTOS_DIR.glob("*.jpg")))
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"获取摄像头状态失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"状态检查失败: {str(e)}"
        }, ensure_ascii=False)

@mcp.tool()
def list_photos() -> str:
    """列出所有已拍摄的照片"""
    try:
        photos = list(PHOTOS_DIR.glob("photo_*.jpg"))
        photos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        photo_list = []
        total_size = 0
        
        for photo in photos[:10]:  # 只显示最近10张照片
            stat = photo.stat()
            size_kb = stat.st_size / 1024
            total_size += size_kb
            
            photo_info = {
                "filename": photo.name,
                "path": str(photo),
                "size": f"{size_kb:.1f} KB",
                "created": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            }
            photo_list.append(photo_info)
        
        return json.dumps({
            "status": "success",
            "total_photos": len(photos),
            "max_photos": camera_manager.max_photos,
            "total_size": f"{total_size:.1f} KB",
            "recent_photos": photo_list
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"获取照片列表失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"获取照片列表失败: {str(e)}"
        }, ensure_ascii=False)
    

# 清理函数 - 程序退出时释放资源
def cleanup():
    """程序退出时的清理操作"""
    camera_manager.release_camera()
    cv2.destroyAllWindows()
    logger.info("摄像头资源清理完成")

if __name__ == "__main__":
    logger.info("启动摄像头拍照识别MCP服务器...")
    
    # 注册退出时的清理操作
    import atexit
    atexit.register(cleanup)
    
    print("摄像头服务器正在运行...")
    mcp.run(transport='stdio')