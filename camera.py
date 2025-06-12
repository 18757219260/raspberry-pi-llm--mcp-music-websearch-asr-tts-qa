import cv2
import base64
import logging
import os
import time
import subprocess
import numpy as np
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from dashscope import MultiModalConversation
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    """树莓派摄像头管理器 - 使用libcamera-vid进行拍照"""
    
    def __init__(self, max_photos=10):
        self.process = None
        self.is_initialized = False
        self.max_photos = max_photos
        
    def initialize_camera(self) -> bool:
        """初始化摄像头，使用libcamera-vid"""
        try:
            # 优化摄像头参数
            cmd = "libcamera-vid -t 0 --width 1280 --height 720 --codec mjpeg --nopreview --framerate 15 -o -"
            
            self.process = subprocess.Popen(
                cmd.split(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=1024*1024
            )
            
            # 等待摄像头启动
            time.sleep(2)
            
            # 检查进程是否正常运行
            if self.process.poll() is None:
                self.is_initialized = True
                logger.info("树莓派摄像头初始化成功")
                return True
            else:
                error_output = self.process.stderr.read().decode()
                logger.error(f"摄像头启动失败: {error_output}")
                return False
                
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False

    def get_frame(self, flush_buffer=True):
        """从libcamera-vid获取单个视频帧"""
        if not self.is_initialized:
            return False, None
            
        buffer = bytearray()
        
        try:
            # 如果需要刷新缓冲区（获取最新帧）
            if flush_buffer:
                # 设置非阻塞模式临时读取并丢弃旧数据
                import fcntl
                import os
                
                # 获取文件描述符
                fd = self.process.stdout.fileno()
                
                # 保存当前标志
                old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                
                # 设置非阻塞模式
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
                
                # 读取并丢弃所有可用数据
                try:
                    while True:
                        chunk = self.process.stdout.read(65536)
                        if not chunk:
                            break
                except (IOError, OSError):
                    # 没有更多数据可读，这是预期的
                    pass
                
                # 恢复阻塞模式
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
                
                # 等待新帧
                time.sleep(0.1)
            
            # 读取数据直到找到完整的JPEG帧
            timeout_start = time.time()
            timeout = 5  # 5秒超时
            
            while time.time() - timeout_start < timeout:
                data = self.process.stdout.read(4096)
                if not data:
                    logger.warning("无法读取摄像头数据")
                    return False, None
                    
                buffer.extend(data)
                
                # 查找JPEG帧的开始和结束标记
                start = buffer.find(b'\xff\xd8')  # JPEG开始标记
                end = buffer.find(b'\xff\xd9', start)  # JPEG结束标记
                
                if start != -1 and end != -1:
                    # 提取完整的JPEG帧
                    jpeg = buffer[start:end + 2]
                    
                    # 解码为OpenCV格式
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        # 翻转帧（如果需要）
                        frame = cv2.flip(frame, 0)
                        return True, frame
                        
            logger.error("获取帧超时")
            return False, None
            
        except Exception as e:
            logger.error(f"获取视频帧失败: {e}")
            return False, None
    
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
            
            # 丢弃一些旧帧，确保获取最新图像
            logger.info("正在获取最新帧...")
            for _ in range(3):
                ret, frame = self.get_frame()
                if not ret:
                    continue
                time.sleep(0.1)
            
            # 获取最终的拍照帧
            ret, frame = self.get_frame()
            if not ret or frame is None:
                return False, "", "无法捕获图像"
            
            # 生成唯一文件名
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
    
    def capture_single_photo(self) -> tuple[bool, np.ndarray, str]:
        """
        拍摄单张照片并返回图像数据（不保存到文件）
        
        Returns:
            tuple: (成功标志, 图像数组, 错误信息)
        """
        if not self.is_initialized:
            if not self.initialize_camera():
                return False, None, "摄像头初始化失败"
        
        try:
            # 丢弃一些旧帧，确保获取最新图像
            for _ in range(3):
                ret, frame = self.get_frame()
                if not ret:
                    continue
                time.sleep(0.1)
            
            # 获取最终的拍照帧
            ret, frame = self.get_frame()
            if not ret or frame is None:
                return False, None, "无法捕获图像"
            
            return True, frame, ""
                
        except Exception as e:
            logger.error(f"拍照过程出错: {e}")
            return False, None, f"拍照失败: {str(e)}"
    
    def release_camera(self):
        """释放摄像头资源"""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
                logger.info("摄像头进程已终止")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logger.warning("强制终止摄像头进程")
            except Exception as e:
                logger.error(f"释放摄像头资源失败: {e}")
            finally:
                self.process = None
                self.is_initialized = False


camera_manager = CameraManager(max_photos=10)

def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64格式"""
    try:
        # 验证文件路径
        if not os.path.exists(image_path):
            logger.error(f"图片文件不存在: {image_path}")
            return ""
        
        # 获取文件信息
        file_stat = os.stat(image_path)
        
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        logger.error(f"图片编码失败 {image_path}: {e}")
        return ""

def encode_frame_to_base64(frame: np.ndarray) -> str:
    """将OpenCV帧编码为base64格式"""
    try:
        # 编码为JPEG格式
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            return ""
        
        # 转换为base64
        encoded_string = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"
        
    except Exception as e:
        logger.error(f"帧编码失败: {e}")
        return ""

def analyze_image_with_qwen(image_data: str, prompt: str = None) -> str:
    """使用通义千问视觉模型分析图片"""
    try:
        # 简化默认提示词
        if prompt is None:
            prompt = "用聊天的方式根据问题来简洁描述图片内容"
        
        if not image_data:
            return "图片编码失败"
        
        # 构建消息内容
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {"image": image_data}
                ]
            }
        ]
        
        # 调用通义千问视觉模型
        response = MultiModalConversation.call(
            model='qwen-vl-max',
            messages=messages,
            api_key='sk-08bb8f6bf6ad4bbd9f33913fb6b6e248'
        )
        
        if response.status_code == 200:
            result = response.output.choices[0].message.content
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
        
        # 使用简化的提示词
        if prompt is None:
            prompt = "用聊天的方式根据问题来简洁描述图片内容"
        
        # 编码图片并分析
        base64_image = encode_image_to_base64(photo_path)
        analysis_result = analyze_image_with_qwen(base64_image, prompt)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"拍照分析失败: {e}")
        return "拍照分析失败"

@mcp.tool()
def take_photo_and_analyze_direct(prompt = None) -> str:
    """直接拍照并分析（不保存到文件，更快速）"""
    try:
        # 直接拍摄照片获取帧数据
        success, frame, error_msg = camera_manager.capture_single_photo()
        
        if not success:
            return f"拍照失败: {error_msg}"
        
        # 使用简化的提示词
        if prompt is None:
            prompt = "用聊天的方式根据问题来简洁描述图片内容"
        
        # 直接编码帧并分析
        base64_image = encode_frame_to_base64(frame)
        if not base64_image:
            return "图片编码失败"
            
        analysis_result = analyze_image_with_qwen(base64_image, prompt)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"直接拍照分析失败: {e}")
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
        
        base64_image = encode_image_to_base64(photo_path)
        analysis_result = analyze_image_with_qwen(base64_image, prompt)
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
            "camera_type": "libcamera (树莓派)",
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
    logger.info("摄像头资源清理完成")

if __name__ == "__main__":
    logger.info("启动树莓派摄像头拍照识别MCP服务器...")
    
    # 注册退出时的清理操作
    import atexit
    atexit.register(cleanup)
    
    print("摄像头服务器正在运行...")
    mcp.run(transport='stdio')