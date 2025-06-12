# 树梅派 LLM 



---

## 目录

- [功能](#功能)
- [配置](#配置)

---

## 系统功能
- **自我设计**<br>
   根据自己需要构建知识库成为量身定做的问答助手  
- **人脸识别**  
  `opencv` 人脸识别

- **语音输入与输出**  
   `百度语音识别API` 和 `edge_tts` 实时语音识别、流式语音输出
- **大模型** 
  `qwenAPI` 和本地向量化模型 `text2vec_base_chinese_q8`知识库检索配合大模型回答
  `mcp`接入`音乐播放`和`网络搜索`和`图像识别`功能：<br>
  支持播放 `继续播放` `下一首` `上一首` `暂停`<br>
  支持查看各地各时间点`天气`，各时间点的地区的`事件`<br>
  支持`实时图像识别`,并根据语义进行不同角度的识别，如`颜色`、`品牌`、`数数`等
  `终端ui`界面




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
mcp-server
pyside6
```

```
首先构建人脸库和知识库
```bash
python mk_faiss.py
python face_create.py 
```

执行以下命令运行主交互程序：

```bash
终端
python main_stream.py 

UI显示
python app.py

```
