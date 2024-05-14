import re
import dashscope
import random
import requests
import json
from http import HTTPStatus
from openai import OpenAI
from time import sleep

from Deployment import Deployer

class LLMTransfer:
    def __init__(self, model_name:str, temperature=10e-4, b_local:bool=False):
        self.model_name = model_name
        self.temperature = temperature
        self.b_local = b_local
        #self.sft_model_path = "/data/zonepg/code/LLaMA-Factory/saves/Qwen1.5-7B-Chat/full/sft/QAs+ds+dialogue/v2"
        self.sft_model_path = "/data/zonepg/code/LLaMA-Factory/saves/Qwen1.5-7B/full/sft/QAs+ds+dialogue/v2"
        if model_name == "sft":
            print("微调模型路径：", self.sft_model_path)
        if b_local:
            self.deployer = Deployer(model_name=model_name, temperature=temperature)

    '''
    阿里系千问大模型调用API
    '''
    def call_with_prompt_qw(self, prompt:str) -> any:
        messages = [{
            'role': 'user', 
            'content': prompt
        }]
        response = dashscope.Generation.call(
            model=self.model_name,
            messages=messages,
            # set the random seed, optional, default to 1234 if not set
            seed=2024,
            result_format='text',  # the format include 'text' and 'message'
            temperature=self.temperature,
        )
        if response.status_code == HTTPStatus.OK:
            return response.output['text']
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
        
        return None
    
    '''
    百度系千帆大模型调用API
    '''
    def call_with_prompt_qf(self, prompt:str) -> any:
        def get_access_token():
            """
            使用 AK，SK 生成鉴权签名（Access Token）
            :return: access_token，或是None(如果错误)
            """
            API_KEY = "jvyY9raEufISxdFTJVN1h229"
            SECRET_KEY = "5bgNjaX1oIETYqOb1d7wNzGsA6ZDyndp"
            url = "https://aip.baidubce.com/oauth/2.0/token"
            params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
            return str(requests.post(url, params=params).json().get("access_token"))
        
        model_name = re.sub('-', '_', self.model_name)
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model_name}?access_token=" + get_access_token()
        
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        sleep(0.5)

        try:
            return json.loads(response.text)["result"]
        except:
            print(f"无效回复：{response.text}")
            return ''

    '''
    调用远程的微调大模型API
    '''
    def call_with_prompt_sft(self, prompt:str) -> any:
        url = {}
        url['qwen1.5-14B 微调'] = 'http://3d8b77a6.r2.cpolar.top/v1'
        url['baichuan13B'] = 'http://517f7e08.r15.vip.cpolar.cn/v1'
        url['qwen14B'] = 'http://3d8b77a6.r2.cpolar.top/v1'
        url['sft_qwen'] = 'https://642a2878.r18.cpolar.top/v1'
        url['sft_baichuan'] = 'http://642a2878.r18.cpolar.top/v1'
        client = OpenAI(base_url="http://llmsapi.vip.cpolar.cn/v1", api_key="sk-coaihv832rfj0qaj09")

        message = [
            {"role": "system", "content": prompt[0]},
            {"role": "user", "content": prompt[1]},
        ]
        
        try:
            response = client.chat.completions.create(
                model = self.sft_model_path,
                messages = message,
                stream=False,
                temperature=self.temperature,
                timeout=60,
                # max_tokens=1024,
            )
        except:
            return "模型回复超时！"

        return response.choices[0].message.content.strip()
    
    '''
    本地大模型调用API
    '''
    def call_with_prompt_local(self, prompt:str) -> str:
        return self.deployer.response(prompt)
    
    def call_with_prompt(self, prompt:str) -> str:
        whole_prompt = prompt[0] + prompt[1]
        # if self.model_name == 'mindchat':
        #     whole_prompt = "请帮我做一个心理学的选择题：\n" + prompt[1]
        # # print(whole_prompt)
        if self.b_local:
            return self.call_with_prompt_local(whole_prompt)
        elif self.model_name in ['qwen1.5-7b-chat', 'qwen1.5-14b-chat', 'baichuan2-13b-chat-v1']:
            return self.call_with_prompt_qw(whole_prompt)
        elif self.model_name in ['yi-34b-chat', 'qianfan-chinese-llama-2-7b', 'qianfan-chinese-llama-2-13b']:
            return self.call_with_prompt_qf(whole_prompt)
        elif self.model_name in ['sft', 'sft-qwen1.5-14b', 'sft-baichuan2-13b']:
            return self.call_with_prompt_sft(prompt)
        else: # baichuan2-7b-chat, chatglm3-6b-32k, chinese-alpaca-2-7b, chinese-alpaca-2-13b
            return self.call_with_prompt_local(whole_prompt)

if __name__ == '__main__':
    llm = LLMTransfer("chinese-alpaca-2-13b")
    result = llm.call_with_prompt_sft('如何做西红柿炒鸡蛋？')
    print(result)
