# 树梅派 LLM 



---

## 目录

- [功能](#功能)
- [配置](#配置)

---

## 系统功能

- **人脸识别**  
  `opencv` 人脸识别

- **语音输入与输出**  
   `百度语音识别API` 和 `edge_tts` 语音识别、语音合成
    `qwenAPI`和本地llama.cpp `text2vec_base_chinese_q8`


---


## 配置
```bash
langchain_community
langchain
langchain_core
llama_cpp
openai
face_recognition
openvcv2
pickle
pyaudio
webrtcvad
baidu-aip
```

```
首先构建人脸库和知识库
```bash
python mk_faiss.py
python face_create.py 
```

执行以下命令运行主交互程序：

```bash

python main_stream.py 


```
