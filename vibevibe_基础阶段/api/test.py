#!/usr/bin/env python3
"""
Minimax API连通性测试脚本
使用OpenAI格式测试
"""

import json
import requests

# 加载配置
CONFIG_FILE = '/Users/bytedance/Documents/trae_projects/vibe_vibe/vibecoding_study/vibevibe_基础阶段/api/config.json'

def load_config():
    """加载配置文件"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return None

def test_openai_format(model_url, api_key, group_id=None):
    """测试OpenAI格式API（Minimax兼容）"""
    print(f"\n📡 正在测试 OpenAI格式 API: {model_url}")
    print(f"🔑 API Key: {api_key[:10]}...{api_key[-10:]}")
    if group_id:
        print(f"👥 Group ID: {group_id}")
    
    # 完整的API端点
    full_url = f"{model_url}/chat/completions"
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Minimax的OpenAI兼容接口需要GroupId请求头
    if group_id:
        headers['GroupId'] = group_id
    
    payload = {
        'model': 'MiniMax-M2.7',
        'messages': [
            {'role': 'system', 'content': '你是一个有用的助手。'},
            {'role': 'user', 'content': '你好，测试一下连接。'}
        ],
        'temperature': 0.7,
        'max_tokens': 100,
        'extra_body': {'reasoning_split': True}
    }
    
    try:
        print("\n⏳ 正在发送请求...")
        print(f"📤 请求URL: {full_url}")
        print(f"📤 请求头: {json.dumps(headers, indent=2)}")
        print(f"📤 请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        response = requests.post(full_url, headers=headers, json=payload, timeout=30)
        print(f"\n📊 HTTP状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ 请求成功！")
            print(f"📋 响应数据:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # 提取回复内容
            if 'choices' in data and len(data['choices']) > 0:
                choice = data['choices'][0]
                content = choice.get('message', {}).get('content', '')
                reasoning_details = choice.get('message', {}).get('reasoning_details', [])
                
                if reasoning_details:
                    thinking_text = reasoning_details[0].get('text', '')
                    print(f"\n💭 思考内容:\n{thinking_text}")
                
                if content:
                    print(f"\n� AI回复:\n{content}")
            else:
                print("\n⚠️ 响应格式不符合预期")
                
        else:
            print(f"\n❌ 请求失败")
            try:
                error_data = response.json()
                print(f"错误信息: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"错误响应: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 网络请求失败: {e}")

def main():
    print("=" * 60)
    print("    Minimax API 连通性测试工具 (OpenAI格式)")
    print("=" * 60)
    
    config = load_config()
    if not config:
        print("\n❌ 请先在配置页面设置API信息")
        return
    
    model_url = config.get('model_url', '')
    api_key = config.get('api_key', '')
    group_id = config.get('group_id', '')
    
    if not model_url:
        print("❌ 模型URL为空")
        return
    
    if not api_key:
        print("❌ API密钥为空")
        return
    
    print(f"\n📝 当前配置:")
    print(f"   URL: {model_url}")
    print(f"   Key: {api_key[:10]}...{api_key[-10:]}")
    if group_id:
        print(f"   Group ID: {group_id}")
    
    print("\n🔍 使用OpenAI格式测试...")
    test_openai_format(model_url, api_key, group_id)
    
    print("\n" + "=" * 60)
    print("测试完成")

if __name__ == '__main__':
    main()