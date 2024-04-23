from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig, TextStreamer
import os
import torch

device = "cuda"

class Deployer(object):
    '''
    model_name: 
    '''
    def __init__(self, model_name:str, temperature=10e-4) -> None:
        self.name2path = {
            "chatglm3-6b-32k": "/THUDM/chatglm3-6b-32k",
            "qwen1.5-7b": "/Qwen/qwen1.5-7b-chat",
            "qwen1.5-7b-chat": "/Qwen/qwen1.5-7b-chat",
            "qwen1.5-14b-chat-int8": "/Qwen/qwen1.5-14b-chat-gptq-int8",
            "qwen1.5-14b-chat": "/Qwen/qwen1.5-14b-chat-gptq-int8",
            "baichuan2-7b-chat": "/baichuan-inc/baichuan2-7b-chat",
            "chinese-alpaca-2-7b": "/hfl/chinese-alpaca-2-7b"
        }
        self.model_name = model_name
        self.temperature = temperature
        self.model_path = os.path.dirname(os.path.abspath(__file__)) + '/models' + self.name2path[self.model_name]
        if self.model_name in ['qwen1.5-7b', 'qwen1.5-7b-chat', "qwen1.5-14b-chat"]:
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
        elif self.model_name in ['baichuan2-7b-chat', 'baichuan2-13b-chat']:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                device_map="auto", 
                torch_dtype="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
            ).half().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )

    def response(self, prompt:str) -> str:
        if self.model_name in ['chatglm3-6b-32k']:
            model = self.model.eval()
            response, history = model.chat(self.tokenizer, prompt, history=[], temperature=self.temperature)
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
            self.model.generation_config = GenerationConfig.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + self.name2path[self.model_name])
            self.model.generation_config.temperature = self.temperature
            messages = [{"role": "user", "content": prompt}]
            response = self.model.chat(self.tokenizer, messages)
        elif self.model_name in ['chinese-alpaca-2-7b']:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
            generate_ids = self.model.generate(
                inputs.input_ids, do_sample=True, max_new_tokens=1024, top_k=10, top_p=0.1, temperature=self.temperature, repetition_penalty=1.18,
                eos_token_id=2, bos_token_id=1, pad_token_id=0, typical_p=1.0,encoder_repetition_penalty=1,
            )
            response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response


if __name__ == '__main__':
    model_path = os.path.dirname(os.path.abspath(__file__)) + "/hfl/chinese-alpaca-2-7b"
    prompt = "你是一位专业的心理咨询助手，拥有丰富的心理学知识。现在有一道心理学知识的多项选择题，需要你利用自己的心理学知识进行解答\n选项中有多个是正确答案，请从ABCD四个选项中选出你认为正确的选项。【题目：躯体神经⽀配的器官包括（）。\n选项：{'A': '内脏器官', 'B': '腺体器官', 'C': '感觉器官', 'D': '运动器官'}】\n请不要质疑题目的正确性。如果你认为题干表述不清晰，请结合自己的心理学知识，给出自己的答案。\n答案（仅回复相应的字母编号）："
    model_name = "chinese-alpaca-2-7b"
    deployer = Deployer(model_name)
    print(deployer.response(prompt))
