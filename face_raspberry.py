import sys
import os
sys.path.append(os.path.abspath("/home/joe/chatbox"))
import cv2
import face_recognition
import numpy as np
import time
import pickle
import subprocess
import os

class FaceRecognizer:
    def __init__(self,timeout=10,face_model="face/face_model.pkl",model_file = 'face_track/models/deploy.prototxt',weights_file = 'face_track/models/res10_300x300_ssd_iter_140000.caffemodel'):
        self.known_faces = {}
        self.known_face_names = []
        self.known_face_encodings = []
        self.process = None
        self.start_time = time.time() 
        self.frame_count = 0 
        self.recognized_user = None  # 修正变量名与获取方法一致
        self.timeout = timeout
        self.face_model = face_model
        self.model_file = model_file
        self.weights_file = weights_file
        
        self._load_known_faces()
        self.use_dnn = self._check_dnn_models()
        self._init_camera()


    def _check_dnn_models(self):
        """检查是否存在DNN模型文件"""
        
        
        print("找到DNN模型文件，尝试加载...")
        self.face_net = cv2.dnn.readNet(self.model_file, self.weights_file)
        print("成功加载DNN模型")
        return True
            
    def _load_known_faces(self):
        """加载人脸数据库"""
        try:
            with open(self.face_model, "rb") as f:
                self.known_faces = pickle.load(f)
                self.known_face_names = list(self.known_faces.keys())
                self.known_face_encodings = list(self.known_faces.values())
                print("成功加载人脸数据库！")
        except FileNotFoundError:
            print("未找到人脸数据库，请先运行创建数据库脚本！")
            exit()

    def _init_camera(self):
        """初始化摄像头，使用libcamera-vid"""
        # 优化摄像头参数，降低帧率减轻处理负担
        cmd = "libcamera-vid -t 0 --width 320 --height 240 --codec mjpeg --nopreview --framerate 15 -o -"
        try:
            self.process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, bufsize=1024*1024)
            print("摄像头初始化成功")
        except Exception as e:
            print(f"初始化摄像头失败: {e}")
            exit()

    def get_frame(self):
        """从libcamera-vid获取视频帧"""
        buffer = bytearray()
        try:
            while True:
                data = self.process.stdout.read(1024)
                if not data:
                    print("无法读取视频数据")
                    return False, None
                buffer.extend(data)
                start = buffer.find(b'\xff\xd8')
                end = buffer.find(b'\xff\xd9', start)
                if start != -1 and end != -1:
                    jpeg = buffer[start:end + 2]
                    buffer = buffer[end + 2:]
                    # 直接解码为BGR格式，减少转换开销
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        print("解码帧失败")
                        continue
                    frame = cv2.flip(frame, 0)
                    # 根据后续处理需要的格式返回
                    if self.use_dnn:
                        return True, frame  
                    else:
                        return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                    
        except Exception as e:
            print(f"获取视频帧失败: {e}")
            return False, None

    def detect_faces_dnn(self, frame):
        """使用DNN模型检测人脸"""
        try:
            h, w = frame.shape[:2]
            # 创建blob并进行前向传播
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            face_locations = []
            
            # 检测结果处理
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # 置信度阈值
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    # 确保坐标在图像范围内
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:  # 有效的人脸区域
                        face_locations.append((y1, x2, y2, x1))  # top, right, bottom, left
            
            return face_locations
        except Exception as e:
            print(f"DNN人脸检测失败: {e}")
            return []

    def recognize_faces(self, frame):
        """执行人脸识别"""
        try:
            # 使用DNN或face_recognition进行人脸检测
            if self.use_dnn:
                face_locations = self.detect_faces_dnn(frame)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame  # face_recognition已经使用RGB格式
                face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=0)
            
            # 跳过小人脸，减少计算量
            valid_face_locations = []
            for loc in face_locations:
                top, right, bottom, left = loc
                face_size = (bottom - top) * (right - left)
                if face_size > 1000:  # 面积阈值
                    valid_face_locations.append(loc)
            
            # 如果没有有效人脸，提前返回
            if not valid_face_locations:
                return []
            
            # 对每个人脸计算特征编码
            start = time.time()
            face_encodings = face_recognition.face_encodings(rgb_frame, valid_face_locations, model="small")
            
            results = []
            for (top, right, bottom, left), face_encoding in zip(valid_face_locations, face_encodings):
                # 只在模型加载后才进行比较
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.45
                    )
                    name = "who?"
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    if True in matches:
                        best_match_index = face_distances.argmin()
                        if face_distances[best_match_index] < 0.4:
                            name = self.known_face_names[best_match_index]
                            end = time.time()
                            elapsed = end - start
                            # print(f"识别到你啦 {name}! 耗时 {elapsed:.3f} 秒")
                else:
                    name = "who?"
                    
                results.append(((top, right, bottom, left), name))
            
            # print(f"检测到 {len(results)} 张人脸")
            return results
        except Exception as e:
            print(f"人脸识别失败: {e}")
            return []

  

    def release(self):
        """释放资源"""
        try:
            if self.process:
                self.process.terminate()
                self.process.wait(timeout=2)
                print("摄像头进程已终止")
        except Exception as e:
            print(f"释放资源失败: {e}")
        cv2.destroyAllWindows()
        print("所有资源已释放")
        
    def main(self, save_path):
        """主函数"""
        
        print("📸 开始人脸认证...")
        start_time = time.time()
        fps_start_time = time.time()
        fps_frame_count = 0
        auth_success = False

        try:
            
            while time.time() - start_time < self.timeout:
                ret, frame = self.get_frame()
                if not ret:
                    print("无法获取视频帧")
                    break

                self.frame_count += 1
                fps_frame_count += 1
                
                # 显示FPS
                if fps_frame_count >= 30:
                    current_time = time.time()
                    elapsed = current_time - fps_start_time
                    fps = fps_frame_count / elapsed
                    print(f"当前FPS: {fps:.2f}")
                    fps_frame_count = 0
                    fps_start_time = current_time
                
                # 跳帧处理，减轻CPU负担
                if self.frame_count % 2 != 0:  # 每隔一帧处理一次
                    continue
                
                # 准备显示帧
                if self.use_dnn:
                    display_frame = frame.copy()
                else:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 识别人脸
                results = self.recognize_faces(frame)
                # 绘制所有人脸框
                for (top, right, bottom, left), name in results:
                    if name != "who?":
                        color = (0, 255, 0)
                        self.recognized_user = name
                        auth_success = True
                        # print(f"✅ 已识别用户: {name}")  
                    else:
                        color=(0, 0, 255)

                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(display_frame, name, (left, top-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                
                # 保存当前帧
                cv2.imwrite(save_path, display_frame)
                if auth_success:
                    print(f"✅人脸认证成功，用户: {self.recognized_user}")
                    break
            if not auth_success:
                print("❌ 认证超时，未能识别用户")
            
        except KeyboardInterrupt:
            print("用户中断程序")
        except Exception as e:
            print(f"程序异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.release()
            cv2.destroyAllWindows()

        print("程序已结束")
        return auth_success,self.recognized_user


if __name__ == "__main__":

    save_path = "frame.jpg"
    recognizer = FaceRecognizer()
    recognizer.main(save_path)