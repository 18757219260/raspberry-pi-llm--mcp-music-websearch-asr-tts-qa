
import requests
import json
import logging
from mcp.server.fastmcp import FastMCP
import subprocess
import tempfile
import os
import time
import signal
from io import BytesIO
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 定义全局播放状态变量
playing = False
pause = False
current_song = None
current_playlist = []
current_index = 0
player_process = None
player_lock = threading.Lock()

# 创建MCP服务器实例
mcp = FastMCP("netease-music")

# 设置 XDG_RUNTIME_DIR 环境变量 (从原始文件中获取)
xdg_runtime_dir = '/run/user/{}'.format(os.getuid())
os.environ['XDG_RUNTIME_DIR'] = xdg_runtime_dir

def stop_current_playback():
    """停止当前播放的mpg123进程"""
    global player_process, playing, pause
    
    with player_lock:
        if player_process is not None:
            try:
                player_process.terminate()
                player_process.wait(timeout=2)
            except Exception as e:
                logger.error(f"停止播放进程失败: {e}")
                try:
                    player_process.kill()
                except:
                    pass
            player_process = None
            playing = False
            pause = False

@mcp.tool()
def search_music(keyword: str, limit: int = 10) -> str:
    """
    搜索音乐
    
    params: 
        keyword: 歌曲名或关键字
        limit: 返回结果数量限制
    """
    try:
        url = f'http://music.163.com/api/search/get/web?csrf_token=hlpretag=&hlposttag=&s={keyword}&type=1&offset=0&total=true&limit={limit}'
        res = requests.get(url)
        music_json = json.loads(res.text)
        
        # 更新全局播放列表
        global current_playlist, current_index
        current_playlist = []
        current_index = 0
        
        count = music_json["result"]["songCount"]
        results = []
        
        if count > 0:
            for i in range(min(count, limit)):
                song = music_json["result"]["songs"][i]
                song_info = {
                    'id': song['id'],
                    'name': song['name'],
                    'artist': ', '.join([ar['name'] for ar in song['artists']]),
                    'album': song['album']['name']
                }
                current_playlist.append(song_info)
                results.append(f"{i+1}. {song_info['name']} - {song_info['artist']}")
            
            return "找到以下歌曲:\n" + "\n".join(results)
        else:
            return "没有找到相关歌曲"
            
    except Exception as e:
        logger.error(f"搜索歌曲失败: {e}")
        return f"搜索出错: {str(e)}"

@mcp.tool()
def play_music(song_name: str) -> str:
    """
    播放音乐 
    
    params: 
        song_name: 歌曲名或关键字
    """
    global playing, pause, current_song, current_playlist, current_index, player_process
    
    try:
        # 搜索歌曲
        url = f'http://music.163.com/api/search/get/web?csrf_token=hlpretag=&hlposttag=&s={song_name}&type=1&offset=0&total=true&limit=10'
        res = requests.get(url)
        music_json = json.loads(res.text)
        
        # 验证搜索结果
        if "result" not in music_json:
            return "搜索请求失败，请稍后重试"
            
        result = music_json["result"]
        count = result.get("songCount", 0)
        songs = result.get("songs", [])
        
        if count == 0 or not songs:
            return f"没有找到与【{song_name}】相关的歌曲"
            
        # 更新播放列表
        current_playlist = []
        for song in songs:
            if all(key in song for key in ['id', 'name', 'artists']):
                song_info = {
                    'id': song['id'],
                    'name': song['name'],
                    'artist': ', '.join([ar['name'] for ar in song['artists']]),
                    'album': song.get('album', {}).get('name', '未知专辑')
                }
                current_playlist.append(song_info)
        
        if not current_playlist:
            return "处理搜索结果时发生错误，无有效歌曲"
            
        # 设置当前播放索引
        current_index = 0
        target_song = current_playlist[0]
        song_id = target_song["id"]
        song_name = target_song["name"]
        artist_name = target_song["artist"]
        
        # 获取音乐URL
        music_url = f'http://music.163.com/song/media/outer/url?id={song_id}.mp3'
        response = requests.get(music_url)
        
        if response.status_code != 200:
            return f"无法获取【{song_name}】的播放链接"
        
        # 修正：正确验证音频数据
        audio_data = response.content
        if len(audio_data) == 0:
            return f"【{song_name}】的音频文件为空，可能无版权或链接失效"
        
        # 创建临时文件
        temp_file_name = "temp_audio.mp3"
        with open(temp_file_name, 'wb') as temp_file:
            temp_file.write(audio_data)
        
        # 停止当前播放
        stop_current_playback()
        
        # 开始播放新歌曲
        try:
            with player_lock:
                player_process = subprocess.Popen(
                    ["mpg123", "-q", temp_file_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # 延迟确保进程启动
                time.sleep(0.2)
                
                # 验证进程是否成功启动
                if player_process.poll() is not None:
                    playing = False
                    pause = False
                    return f"音乐播放器启动失败"
                
                playing = True
                pause = False
                current_song = target_song
            
            return 
            
        except Exception as e:
            playing = False
            pause = False
            logger.error(f"播放进程启动失败: {e}")
            return f"播放失败: {str(e)}"
        
    except Exception as e:
        playing = False
        pause = False
        logger.error(f"播放音乐过程中出错: {e}", exc_info=True)
        return f"播放音乐时发生错误: {str(e)}"

@mcp.tool()
def play_by_index(index: int) -> str:
    """
    播放指定索引的歌曲
    
    params:
        index: 歌曲在播放列表中的索引(从1开始)
    """
    global playing, pause, current_song, current_playlist, current_index, player_process
    
    if not current_playlist:
        return "播放列表为空，请先搜索歌曲"
    
    try:
        # 调整为0-based索引
        idx = index - 1
        if idx < 0 or idx >= len(current_playlist):
            return f"索引超出范围，当前播放列表有 {len(current_playlist)} 首歌曲"
        
        song = current_playlist[idx]
        song_id = song['id']
        
        # 获取音乐URL并下载
        music_url = f'http://music.163.com/song/media/outer/url?id={song_id}.mp3'
        response = requests.get(music_url)
        audio_data = BytesIO(response.content)
        
        # 创建临时文件
        temp_file_name = "temp_audio.mp3"
        with open(temp_file_name, 'wb') as temp_file:
            temp_file.write(audio_data.getbuffer())
        
        # 停止当前播放
        stop_current_playback()
        
        # 开始播放新歌曲，使用mpg123
        with player_lock:
            player_process = subprocess.Popen(
                ["mpg123", "-q", temp_file_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            playing = True
            pause = False
            current_song = song
            current_index = idx
        
        return  
    
    except Exception as e:
        logger.error(f"播放失败: {e}")
        return f"播放失败: {str(e)}"

@mcp.tool()
def isPlaying() -> str:
    """
    检查是否正在播放
    
    return: 
        playing 如果正在播放，not playing 如果没有播放
    """
    # 检查进程是否还在运行
    global player_process, playing
    
    with player_lock:
        if player_process is not None:
            # 检查进程是否还在运行
            if player_process.poll() is None:
                if playing:  # 不是暂停状态
                    return "playing"
            else:
                # 进程已结束
                player_process = None
                playing = False
                
    return "not playing"

@mcp.tool()
def stopplay() -> str:
    """
    停止播放音乐
    
    返回: 
        播放状态: 已停止
    """
    if playing or pause:
        stop_current_playback()
        return "已停止。"
    else:
        return "当前没有播放中的音乐"

@mcp.tool()
def pauseplay() -> str:
    """
    暂停音乐播放
    
    返回: 
        播放状态: 已暂停
    """
    global playing, pause, player_process
    
    with player_lock:
        if playing and player_process is not None:
            try:
                # 对mpg123发送SIGSTOP信号暂停播放
                player_process.send_signal(signal.SIGSTOP)
                playing = False
                pause = True
                return "已暂停。"
            except Exception as e:
                logger.error(f"暂停播放失败: {e}")
                return f"暂停失败: {str(e)}"
        else:
            return "当前没有播放中的音乐"

@mcp.tool()
def unpauseplay() -> str:
    """
    恢复音乐播放
    
    返回：
        播放状态: 已恢复播放
    """
    global playing, pause, player_process
    
    with player_lock:
        if pause and player_process is not None:
            try:
                # 对mpg123发送SIGCONT信号恢复播放
                player_process.send_signal(signal.SIGCONT)
                playing = True
                pause = False
                return "已恢复播放"
            except Exception as e:
                logger.error(f"恢复播放失败: {e}")
                return f"恢复失败: {str(e)}"
        else:
            return "没有暂停的音乐可以恢复"

@mcp.tool()
def next_song() -> str:
    """
    播放下一首歌曲
    
    返回：
        播放状态
    """
    global current_index, current_playlist
    
    if not current_playlist:
        return "播放列表为空"
    
    if current_index < len(current_playlist) - 1:
        return play_by_index(current_index + 2)  
    else:
        return "已经是最后一首歌曲"

@mcp.tool()
def previous_song() -> str:
    """
    播放上一首歌曲
    
    返回：
        播放状态
    """
    global current_index, current_playlist
    
    if not current_playlist:
        return "播放列表为空"
    
    if current_index > 0:
        return play_by_index(current_index)  # 因为play_by_index接受1-based索引
    else:
        return "已经是第一首歌曲"

@mcp.tool()
def get_playlist() -> str:
    """
    获取当前播放列表
    
    返回：
        播放列表信息
    """
    if not current_playlist:
        return "播放列表为空"
    
    result = "当前播放列表:\n"
    for i, song in enumerate(current_playlist):
        prefix = "▶ " if i == current_index and playing else "  "
        result += f"{prefix}{i+1}. {song['name']} - {song['artist']}\n"
    
    return result

# 添加一个清理函数，确保退出时停止所有播放
def cleanup():
    stop_current_playback()

if __name__ == "__main__":
    logger.info("启动网易云音乐MCP服务器...")
    # 注册退出时的清理操作
    import atexit
    atexit.register(cleanup)
    print("Server running")
    mcp.run(transport='stdio')