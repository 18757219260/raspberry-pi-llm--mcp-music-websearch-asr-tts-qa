import time
import json
import asyncio
from datetime import datetime
from collections import deque
import logging

class ConversationManager:
    def __init__(self, max_history=10, tracking_file="conversation_tracking.json"):
        # 对话历史管理
        self.conversation_history = deque(maxlen=max_history)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 性能跟踪指标
        self.tracking_data = {
            'session_id': self.session_id,
            'user_id': None,
            'start_time': time.time(),
            'total_questions': 0,
            'total_responses': 0,
            'avg_response_time': 0,
            'response_times': [],
            'error_count': 0,
            'conversation_log': []
        }
        
        self.tracking_file = tracking_file
        self._lock = asyncio.Lock()
    
    async def add_conversation_entry(self, question, answer, response_time=None):
        """异步添加对话记录"""
        async with self._lock:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer,
                'response_time': response_time
            }
            
            # 添加到历史记录
            self.conversation_history.append(entry)
            
            # 更新跟踪数据
            self.tracking_data['total_questions'] += 1
            self.tracking_data['total_responses'] += 1
            
            if response_time:
                self.tracking_data['response_times'].append(response_time)
                self.tracking_data['avg_response_time'] = sum(self.tracking_data['response_times']) / len(self.tracking_data['response_times'])
            
            # 轻量级日志记录（仅保存关键信息）
            log_entry = {
                'time': datetime.now().strftime("%H:%M:%S"),
                'q': question[:50] + '...' if len(question) > 50 else question,
                'response_time': round(response_time, 2) if response_time else None
            }
            self.tracking_data['conversation_log'].append(log_entry)
    
    async def record_error(self, error_type, error_message):
        """记录错误信息"""
        async with self._lock:
            self.tracking_data['error_count'] += 1
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': error_type,
                'message': str(error_message)[:100]
            }
            
            if 'errors' not in self.tracking_data:
                self.tracking_data['errors'] = []
            self.tracking_data['errors'].append(error_entry)
    
    def get_conversation_context(self, max_context=3):
        """获取最近的对话上下文"""
        recent_conversations = list(self.conversation_history)[-max_context:]
        context = []
        for conv in recent_conversations:
            context.append(f"qustion: {conv['question']}")
            context.append(f"answer: {conv['answer']}")
        return "\n".join(context)
    
    async def save_tracking_data(self):
        """异步保存跟踪数据"""
        async with self._lock:
            try:
                with open(self.tracking_file, 'w', encoding='utf-8') as f:
                    json.dump(self.tracking_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.error(f"保存跟踪数据失败: {e}")
    
    def get_session_summary(self):
        """获取会话摘要"""
        duration = time.time() - self.tracking_data['start_time']
        return {
            'session_id': self.session_id,
            'duration': round(duration, 2),
            'total_questions': self.tracking_data['total_questions'],
            'avg_response_time': round(self.tracking_data['avg_response_time'], 2),
            'error_count': self.tracking_data['error_count']
        }