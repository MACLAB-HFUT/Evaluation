from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig
import os
import torch

device = "cuda"

class Deployer(object):
    '''
    model_name: 
    '''
    def __init__(self, model_name:str, temperature=10e-4) -> None:
        self.name2path = {
            "chatglm3-6b-32k": "THUDM/chatglm3-6b-32k",
            "qwen1.5-7b": "Qwen/qwen1.5-7b-chat",
            "qwen1.5-7b-chat": "Qwen/qwen1.5-7b-chat",
            "qwen1.5-14b-chat-int8": "Qwen/qwen1.5-14b-chat-gptq-int8",
            "qwen1.5-14b-chat": "Qwen/qwen1.5-14b-chat-gptq-int8",
            "baichuan2-7b-chat": "baichuan-inc/baichuan2-7b-chat",
            "chinese-alpaca-2-7b": "hfl/chinese-alpaca-2-7b",
            "yi-6b-chat": "01-ai/yi-6b-chat",
            "soulchat": "scutcyr/soulchat-chatglm-6b",
            "mindchat": "X-D-Lab/mindchat-qwen-7b-v2"
        }
        self.model_name = model_name
        self.temperature = temperature
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", self.name2path[self.model_name])
        if self.model_name in ['qwen1.5-7b', 'qwen1.5-7b-chat', "qwen1.5-14b-chat", "mindchat"]:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto"
            ).half().cuda()
        elif self.model_name in ['qwen1.5-14b-chat-int8', "chinese-alpaca-2-7b"]:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="cuda:0"
            )
        elif self.model_name in ['baichuan2-7b-chat', 'baichuan2-13b-chat', 'yi-6b-chat']:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                device_map="auto", 
                torch_dtype="auto",
                trust_remote_code=True
            )
        elif self.model_name in ['soulchat']:
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                device_map="auto",
                torch_dtype='auto'
            ).half().cuda()
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                device_map="auto",
                torch_dtype='auto'
            ).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
        )

    def response(self, prompt:str) -> str:
        if self.model_name in ['chatglm3-6b-32k', 'soulchat', 'mindchat']:
            if self.model_name == 'soulchat':
                prompt = "用户：" + prompt + "\n心理咨询师："
            response, history = self.model.chat(self.tokenizer, prompt, history=[], temperature=self.temperature)
        elif self.model_name in ['qwen1.5-7b', 'qwen1.5-7b-chat', "qwen1.5-14b-chat-int8"]:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=self.temperature
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elif self.model_name in ['baichuan2-7b-chat', 'baichuan2-13b-chat']:
            self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
            self.model.generation_config.temperature = self.temperature
            messages = [{"role": "user", "content": prompt}]
            response = self.model.chat(self.tokenizer, messages)
        elif self.model_name in ['chinese-alpaca-2-7b']:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
            generate_ids = self.model.generate(
                inputs.input_ids, do_sample=True, max_new_tokens=1024, temperature=self.temperature, repetition_penalty=1.18,
                eos_token_id=2, bos_token_id=1, pad_token_id=0, typical_p=1.0,encoder_repetition_penalty=1,
            )
            response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        elif self.model_name in ['yi-6b-chat']:
            messages = [
                {"role": "user", "content": prompt}
            ]
            input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
            output_ids = self.model.generate(input_ids.to('cuda'), max_length=256)
            response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return response


if __name__ == '__main__':
    # model_path = os.path.dirname(os.path.abspath(__file__)) + "/hfl/chinese-alpaca-2-7b"
    prompt = "你是一位专业的心理咨询助手，拥有丰富的心理学知识。现在有一道心理学知识的多项选择题，需要你利用自己的心理学知识进行解答\n选项中有多个是正确答案，请从ABCD四个选项中选出你认为正确的选项。【题目：躯体神经⽀配的器官包括（）。\n选项：{'A': '内脏器官', 'B': '腺体器官', 'C': '感觉器官', 'D': '运动器官'}】\n你的答案是："
    # prompt = "你好，请你帮我做一道心理学方面的选择题：\n题目：躯体神经⽀配的器官包括（）。\n选项：{'A': '内脏器官', 'B': '腺体器官', 'C': '感觉器官', 'D': '运动器官'}\n你的答案是："
    model_name = "mindchat"
    deployer = Deployer(model_name)
    print(deployer.response(prompt))
