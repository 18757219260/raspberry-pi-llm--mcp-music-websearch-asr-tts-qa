import numpy as np
import os
import pickle
import face_recognition

# 配置已知人脸数据集，每个名字对应单张或多张图片路径
# known_faces = {
    # "wudawang": ["face_recognition/images/wudawang/1.jpg", "face_recognition/images/wudawang/2.jpg"],  # 多张图片
    # "jiazhuo": ["face_recognition/images/jiazhuo/1.jpg",  "face_recognition/images/jiazhuo/2.jpg"], 
    # "yuhui": "face_recognition/images/yuhui.jpg",
    # "yihao": "face_recognition/images/yihao.jpg",  
    # "yianhao": "face_recognition/images/jianhao.jpg"}

def train_face_model():
    Encodings = []
    Names = []
    
    # 遍历已知人脸数据集
    for name, image_paths in known_faces.items():
        if isinstance(image_paths, str):  # 单张图片
            image_paths = [image_paths]  # 转换为列表
        
        all_encodings = []
        
        # 遍历所有图片路径进行特征提取
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"Warning: {image_path} not found!")
                continue
            
            # 加载图片并提取特征
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                print(f"No face found in {image_path}!")
                continue
            
            # 提取人脸特征编码
            encoding = face_recognition.face_encodings(image, face_locations)[0]
            all_encodings.append(encoding)
        
        # 如果提取了多张图片的特征，则取特征的平均值
        if all_encodings:
            average_encoding = np.mean(all_encodings, axis=0)
            Encodings.append(average_encoding)
            Names.append(name)
            print(f"Processed {name} successfully")
        else:
            print(f"No valid encodings found for {name}")
    
    # 将人名和特征一一对应存入字典中
    face_database = dict(zip(Names, Encodings))

    # 保存训练模型
    with open('face_recognition/face_model.pkl', 'wb') as f:
        pickle.dump(face_database, f)  # 保存字典
    print("Model training completed")

if __name__ == "__main__":
    train_face_model()
