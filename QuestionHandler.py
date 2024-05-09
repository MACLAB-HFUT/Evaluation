import re
import os
import json

from LLMTransfer import LLMTransfer

class QuestionHandler(object):

    '''
    This class is used to handle the question and answer files.
    question_type: single, multiple, mix, essay
    '''
    def __init__(self, question_files:list, answer_files:list, question_type:str) -> None:
        self.question_files = question_files
        self.answer_files = answer_files
        self.question_type = question_type
    
    def prompt_generation(self):
        pass


class GeneralQuestionHandler(QuestionHandler):

    def __init__(self, question_files:list, answer_files:list, question_type:str) -> None:
        # super().__init__(question_files, answer_files, question_type) 
        self.question_files = question_files
        self.answer_files = answer_files
        self.question_type = question_type
        self.contents = self.content_processing()
        self.questions = self.question_processing()
        self.true_answers = self.true_answer_processing() 
    
    def content_processing(self):
        contents = []
        for question_file in self.question_files:
                f = open(question_file, 'r', encoding='utf-8')
                content = json.load(f)
                contents += content
        return contents

    def question_processing(self):
        questions = []
        if self.question_type != 'essay':
            questions = [content['question'] for content in self.contents]
        else:
            questions = [content['questions_and_answers'] for content in self.contents]
            # todo
        return questions

    def true_answer_processing(self):
        answers = []
        if self.question_type != 'essay':
            answers = [content['answer'].strip() for content in self.contents]
        else:
            answers = [content['questions_and_answers'] for content in self.contents]
            # todo
        return answers

    def model_answer_processing(self, response):
        # 正则匹配表达式
        if self.question_type == 'single':
            answer_rule = r"[ABCD](?!.*[ABCD])"
        else:
            answer_rule = r"[ABCD]?['\"]?\s*[、,，和]?\s*[选项]*['\"]?[ABCD]?['\"]?\s*[、,，和]?\s*[选项]*['\"]?[ABCD]?['\"]?\s*[、,，和]?\s*[选项]*['\"]?[ABCD](?!.*[ABCD])"
        try:
            answer_str = re.findall(answer_rule, response, re.S)[0]
            answer_list = re.findall(r"[ABCD]+", answer_str)
            answer = ''.join(answer_list)
        except:
            print(f'答案格式不匹配，模型回答：{response}')
            answer = 'E'
        return answer
    
    def prompt_generation(self, question:str, options:str) -> tuple:
        def prefix_prompt_generation(question_type):
            if question_type == 'single':
                type = '单选题'
            elif question_type == 'multiple':
                type = '多选题'
            elif question_type == 'mix':
                type = '不定项选择题'
            else:
                type = '简答题'
            return f'你是一位专业的心理咨询助手，拥有丰富的心理学知识。现在有一道心理学知识的{type}，需要你利用自己的心理学知识进行解答。\n'
        def question_prompt_generation(question_type, question, options=''):
            if question_type == 'single':
                prompt = '只有一个'
            elif question_type == 'multiple':
                prompt = '有多个'
            elif question_type == 'mix':
                prompt = '可能不止一个'
            options = str(options).strip('{}').replace("'", "").replace(",", "\n").replace(":", ".").replace(" ", "").strip()
            return f'**题目**：\n{question}\n**选项**：\n{options}\n选项中{prompt}是正确答案，**请先给出你的解题思路，再从ABCD四个选项中选出你认为正确的选项**。'
        system_prompt = prefix_prompt_generation(self.question_type)
        question_prompt = question_prompt_generation(self.question_type, question, options)
        # answer_prompt = f'你的答案是（仅回复相应的字母编号）：'
        answer_prompt = f'你的答案是：'
        user_prompt = question_prompt + answer_prompt
        
        return system_prompt, user_prompt
    
    def simple_prompt_generation(self, question:str, options:str) -> tuple:
        if self.question_type == 'single':
            type_prompt = '只有一个'
        else:
            type_prompt = '有多个'
        system_prompt = ""
        question_prompt = f"请回答下面这道心理学相关的选择题\n**题目**：\n{question}\n**选项**：\n{options}\n"
        answer_prompt = f"选项中{type_prompt}是正确答案，从ABCD四个选项中选出你认为正确的选项。你的答案是："
        user_prompt = question_prompt + answer_prompt

        return system_prompt, user_prompt
    

class CaseQuestionHandler(QuestionHandler):

    def __init__(self, question_files: list, answer_files: list, question_type: str) -> None:
        # super().__init__(question_files, answer_files, question_type)
        self.question_files = question_files
        self.answer_files = answer_files
        self.question_type = question_type
        self.contents = self.content_processing()
        self.true_answers = self.true_answer_processing()
        self.cases = self.question_processing()

    def content_processing(self):
        contents = []
        for question_file in self.question_files:
            f = open(question_file, 'r', encoding='utf-8')
            content = json.load(f)
            contents += content
        return contents

    def question_processing(self):
        cases = []
        for i, content in enumerate(self.contents):
            case = {}
            case['general_info'] = content['general_info']
            case['case_introduction'] = content['case_introduction']
            if 'options' in content['questions_and_answers'][0].keys():
                case['qa_pair'] = [(qa_pair['question'], qa_pair['answer'], qa_pair['options'], qa_pair['analysis']) for qa_pair in content['questions_and_answers']]
            else:
                case['qa_pair'] = [(qa_pair['question'], qa_pair['answer']) for qa_pair in content['questions_and_answers']]
            cases.append(case)
        return cases
    
    def mix_choice_prompt_generation(self, index:int):
        case = self.cases[index]
        prompts = []
        for i in range(len(case['qa_pair'])):
            options = str(case['qa_pair'][i][2]).strip('{}').replace("'", "").replace(",", "\n").replace(":", ".").replace(" ", "").strip()
            system_prompt = '你是一位专业的心理咨询助手，拥有丰富的心理学知识和心理咨询经验。现在有一道基于心理咨询场景的选择题，需要你利用自己的心理学知识以及心理咨询经验进行解答。\n'
            question_prompt = f"**求助者的个人信息**：\n{case['general_info'].strip()}\n" + f"**心理咨询内容**：\n{case['case_introduction'].strip()}\n" + "请仔细阅读上述个人信息和心理咨询内容，紧扣上述材料，回答下列问题：\n" \
            + f"**题目**：\n{case['qa_pair'][i][0]}\n"  + f"**选项**：\n{options}\n"
            # answer_prompt = '你的答案是（仅回复相应的字母编号）：'
            answer_prompt = '选项中可能不止一个是正确答案，**请先给出你的解题思路，再从ABCD四个选项中选出你认为正确的选项**。你的答案是：'
            user_prompt = question_prompt + answer_prompt
            prompts.append((system_prompt, user_prompt))
        return prompts
    
    def essay_prompt_generation(self, index:int):
        case = self.cases[index]
        prompts = []
        for i in range(len(case['qa_pair'])):
            system_prompt = '你是一位专业的心理咨询助手，拥有丰富的心理学知识和心理咨询经验。现在有一道基于心理咨询场景的简答题，需要你利用自己的心理学知识以及心理咨询经验进行解答。\n'
            question_prompt = f"**求助者的个人信息**：\n{case['general_info'].strip()}\n" + f"**心理咨询内容**：\n{case['case_introduction'].strip()}\n" + "请仔细阅读上述个人信息和心理咨询内容，紧扣上述材料，回答下列问题。\n" + f"**题目**：{case['qa_pair'][i][0].strip()}\n"
            # answer_prompt = '你的答案是（仅回复你的回答）：'
            answer_prompt = '你的答案是：'
            user_prompt = question_prompt + answer_prompt
            prompts.append((system_prompt, user_prompt))
        return prompts
    
    def prompt_generation(self, index:int):
        if self.question_type == "mix":
            return self.mix_choice_prompt_generation(index)
        else:
            return self.essay_prompt_generation(index)
    
    def true_answer_processing(self):
        answers = []
        for content in self.contents:
            answer = []
            for question in content['questions_and_answers']:
                answer.append(question['answer'].strip())
            answers.append(answer)
        return answers

    def model_answer_processing(self, response):
        # 正则匹配表达式
        # answer_rule = r"[ABCD]+(?!.*[ABCD])"
        answer_rule = r"[ABCD]?['\"]?\s*[、,，和]?\s*[选项]*['\"]?[ABCD]?['\"]?\s*[、,，和]?\s*[选项]*['\"]?[ABCD]?['\"]?\s*[、,，和]?\s*[选项]*['\"]?[ABCD](?!.*[ABCD])"
        try:
            answer_str = re.findall(answer_rule, response, re.S)[0]
            answer_list = re.findall(r"[ABCD]+", answer_str)
            answer = ''.join(answer_list)
        except:   
            print(f'答案格式不匹配，模型回答：{response}')
            answer = 'E'
        return answer

    
if __name__ == '__main__':
    answer_files = ['answer.json']
    question_type = 'essay'
    question_files = [os.path.dirname(os.path.abspath(__file__)) + '/questions/cases/mix/third/' + '11-13 案例混合选择.json']
    question_handler = CaseQuestionHandler(question_files, answer_files, question_type)
    prompt = question_handler.mix_choice_prompt_generation(0)
    print(prompt[0][0])

    # rule = r"[ABCD]?['\"]?\s*[、,，和]?\s*[选项]*['\"]?[ABCD]?['\"]?\s*[、,，和]?\s*[选项]*['\"]?[ABCD]?['\"]?\s*[、,，和]?\s*[选项]*['\"]?[ABCD](?!.*[ABCD])"
    # prompt = "Hello world! This is BC a test string containing some 'A'，'B'和 'C,     'D' letters like oifhoifhia foiwqf"
    # # rule = r"([ABCD]?'?[、,和]?'?){3}[ABCD](?!.*[ABCD])"
    # answer = re.findall(rule, '''综上所述，选项B和选项C是正确的答案。''', re.S)
    # options = re.findall(r"[ABCD]+", answer[0])
    # print(answer)
    # print("".join(options))
 