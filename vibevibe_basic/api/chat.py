#!/usr/bin/env python3
"""
AI聊天机器人后端服务
使用大模型API进行智能对话
支持OpenAI格式、Claude格式和Minimax标准格式
"""

import os
import json
import requests
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config():
    """加载配置文件"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config):
    """保存配置文件"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

# 初始化配置
config = load_config()

# 系统提示词 - 让AI扮演巷猫
SYSTEM_PROMPT = """
你是巷猫的数字分身，一个正在学习用AI的研究生。

关于巷猫的信息：
- 名字：巷猫
- 身份：研究生
- 正在做：系统学习vibecoding，整合编程能力和各种爱好
- 兴趣：AI、吉他、篮球、阅读
- 特点：喜欢做系统的笔记，有逻辑地学习
- 最近：搭建个人主页，整理作品集
- 擅长：内容表达、AI应用、知识整理

你需要：
1. 用自然友好的语言回答问题
2. 尽量保持回答简洁
3. 如果问题与巷猫无关，可以适当发挥但不要偏离主题
4. 回答风格要像真人聊天一样自然

常见问题参考：
- 你现在在做什么？ -> 我正在系统学习vibecoding，搭建自己的个人主页
- 你有哪些作品？ -> 我有几个小项目展示在个人主页上，欢迎查看
- 怎么联系你？ -> 你可以通过微信、邮箱或手机联系我
"""

def call_openai_api(message, model_url, api_key, group_id=None):
    """调用OpenAI格式的API（支持Minimax的OpenAI兼容接口）"""
    headers = {
        'Content-Type': 'application/json',
    }
    
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    # Minimax的OpenAI兼容接口需要GroupId请求头
    if group_id:
        headers['GroupId'] = group_id
    
    # 完整的API端点
    full_url = f"{model_url}/chat/completions"
    
    payload = {
        'model': 'MiniMax-M2.7',
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': message}
        ],
        'temperature': 0.7,
        'max_tokens': 1000
    }
    
    response = requests.post(full_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # 处理响应
    if 'choices' in data and len(data['choices']) > 0:
        choice = data['choices'][0]
        message_content = choice.get('message', {}).get('content', '')
        
        # Minimax返回的内容可能包含<think>标签，需要清理
        if message_content:
            # 移除<think>标签及其内容
            import re
            message_content = re.sub(r'<think>[\s\S]*?</think>', '', message_content).strip()
        
        return message_content.strip()
    elif 'response' in data:
        return data['response'].strip()
    elif 'content' in data:
        return data['content'].strip()
    else:
        return f"API返回格式未知: {json.dumps(data, indent=2)}"

def call_minimax_api(message, model_url, api_key):
    """调用Minimax标准格式的API"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Minimax的标准格式
    payload = {
        'model': 'abab6-chat',
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': message}
        ],
        'stream': False,
        'max_tokens': 1024,
        'temperature': 0.7,
        'top_p': 0.95
    }
    
    response = requests.post(model_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    if 'reply' in data:
        return data['reply'].strip()
    elif 'choices' in data and len(data['choices']) > 0:
        return data['choices'][0].get('message', {}).get('content', json.dumps(data)).strip()
    else:
        return f"API返回格式未知: {json.dumps(data, indent=2)}"

def call_claude_api(message, model_url, api_key, group_id=None):
    """调用Claude格式的API（通过Minimax代理）"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Minimax Anthropic格式需要GroupId请求头
    if group_id:
        headers['GroupId'] = group_id
    
    # Minimax Anthropic格式（兼容Anthropic API）
    payload = {
        'model': 'MiniMax-M2.7',
        'max_tokens': 1000,
        'system': SYSTEM_PROMPT,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': message
                    }
                ]
            }
        ],
        'temperature': 0.7
    }
    
    response = requests.post(model_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # 处理Anthropic格式的响应
    if 'content' in data and isinstance(data['content'], list):
        text_parts = []
        for block in data['content']:
            if block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
            elif block.get('type') == 'thinking':
                continue
        return ''.join(text_parts).strip()
    elif 'choices' in data and len(data['choices']) > 0:
        return data['choices'][0].get('message', {}).get('content', json.dumps(data)).strip()
    elif 'completion' in data:
        return data['completion'].strip()
    elif 'reply' in data:
        return data['reply'].strip()
    else:
        return f"API返回格式未知: {json.dumps(data, indent=2)}"

def call_llm_api(message, model_url, api_key, group_id=None):
    """调用大模型API（自动识别类型）"""
    try:
        # 根据URL判断API类型
        if '/anthropic' in model_url.lower():
            # Claude API格式
            return call_claude_api(message, model_url, api_key, group_id)
        elif '/v1' in model_url.lower():
            # OpenAI格式（包括Minimax的/v1接口）
            return call_openai_api(message, model_url, api_key, group_id)
        elif 'minimax' in model_url.lower():
            return call_minimax_api(message, model_url, api_key)
        else:
            # 默认使用OpenAI格式
            return call_openai_api(message, model_url, api_key, group_id)
            
    except requests.exceptions.RequestException as e:
        return f"API调用失败: {str(e)}"

@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message.strip():
            return jsonify({'error': '消息不能为空'}), 400
        
        # 获取配置
        model_url = config.get('model_url', '')
        api_key = config.get('api_key', '')
        group_id = config.get('group_id', '')
        
        if not model_url:
            return jsonify({
                'error': '请先配置模型URL',
                'message': '你还没有配置大模型API，请在配置页面设置模型URL和API密钥'
            }), 400
        
        # 调用大模型
        response = call_llm_api(message, model_url, api_key, group_id)
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """获取当前配置"""
    return jsonify({
        'model_url': config.get('model_url', ''),
        'api_key_set': bool(config.get('api_key', '')),
        'group_id': config.get('group_id', '')
    })

@app.route('/api/config', methods=['POST'])
def set_config():
    """设置配置"""
    try:
        data = request.get_json()
        model_url = data.get('model_url', '').strip()
        api_key = data.get('api_key', '').strip()
        group_id = data.get('group_id', '').strip()
        
        config['model_url'] = model_url
        config['api_key'] = api_key
        config['group_id'] = group_id
        save_config(config)
        
        return jsonify({'success': True, 'message': '配置保存成功'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test():
    """测试接口"""
    return jsonify({'status': 'ok', 'message': '服务运行正常'})

if __name__ == '__main__':
    # 确保目录存在
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    
    # 如果没有配置文件，创建默认配置
    if not os.path.exists(CONFIG_FILE):
        save_config({
            'model_url': '',
            'api_key': '',
            'group_id': ''
        })
    
    app.run(host='0.0.0.0', port=5000, debug=True)