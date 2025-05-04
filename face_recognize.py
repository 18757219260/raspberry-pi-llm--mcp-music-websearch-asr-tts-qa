import cv2
import face_recognition
import numpy as np
import time
import pickle
import os


class FaceRecognizer:
    def __init__(self, face_model_path="face_model.pkl", timeout=10):
        self.face_model_path = face_model_path
        self.timeout = timeout
        self.known_faces = {}
        self.recognized_user = None
        
    def initialize(self):
        """初始化人脸识别器"""
        try:
            self._load_face_model()
            print("✅ 人脸模型加载成功")
            return True
        except Exception as e:
            print(f"❌ 人脸模型加载失败: {e}")
            return False
    
    def _load_face_model(self):
        """加载人脸数据库"""
        try:
            with open(self.face_model_path, "rb") as f:
                self.known_faces = pickle.load(f)
            print(f"成功加载人脸数据库！共有{len(self.known_faces)}个人脸模型")
        except FileNotFoundError:
            print(f"未找到人脸数据库文件: {self.face_model_path}")
            raise

    def recognize_face(self):
        """执行人脸认证过程，返回识别结果和用户名"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return False, None
        
        print("📸 开始人脸认证...")
        
        start_time = time.time()
        auth_success = False
        
        try:
            while time.time() - start_time < self.timeout:
                # 读取一帧视频
                ret, frame = cap.read()
                if not ret:
                    print("无法获取视频帧")
                    break
                
                # 转换为RGB格式用于face_recognition库
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 人脸检测和识别
                face_locations = face_recognition.face_locations(rgb_frame)
                if not face_locations:
                    # 如果没有检测到人脸，显示提示
                    cv2.putText(frame, "No Face Detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # 检测到人脸，执行识别
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # 与已知人脸比较
                        matches = face_recognition.compare_faces(
                            list(self.known_faces.values()), face_encoding, tolerance=0.4
                        )
                        name = "who?"
                        color = (0, 0, 255)  # 红色表示未识别
                        
                        if True in matches:
                            # 找到最佳匹配
                            face_distances = face_recognition.face_distance(
                                list(self.known_faces.values()), face_encoding
                            )
                            best_match_index = face_distances.argmin()
                            
                            if face_distances[best_match_index] < 0.5:
                                name = list(self.known_faces.keys())[best_match_index]
                                color = (0, 255, 0)  # 绿色表示已识别
                                self.recognized_user = name
                                auth_success = True
                                print(f"✅ 已识别用户: {name}")
                        
                        # 在图像上绘制人脸框和名称
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, name, (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                
                # 显示剩余时间
                remaining = int(self.timeout - (time.time() - start_time))
                cv2.putText(frame, f"time: {remaining}s", (10, frame.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                
                # 显示图像
                cv2.imshow('Face recognition', frame)
                
                # 如果认证成功，停止循环
                if auth_success:
                    # 显示成功信息2秒后继续
                    # cv2.putText(frame, f"{name}", (frame.shape[1]//4, frame.shape[0]//2), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('face recognition', frame)
                    cv2.waitKey(2000)  # 显示2秒
                    break
                
                # 按q键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 如果超时未识别
            if not auth_success:
                print("❌ 认证超时，未能识别用户")
        
        finally:
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()
        
        return auth_success, self.recognized_user

    def get_recognized_user(self):
        """获取已识别的用户名"""
        return self.recognized_user


# 简单测试代码
if __name__ == "__main__":
    face_system = FaceRecognizer()
    if face_system.initialize():
        success, user = face_system.recognize_face()
        if success:
            print(f"认证成功！欢迎 {user}")
        else:
            print("认证失败！")