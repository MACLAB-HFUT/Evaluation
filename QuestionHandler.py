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
            # answer_rule = r'(?<=【)[ABCD](?=】)'
            answer_rule = r'[ABCD]$'
        else:
            # answer_rule = r'(?<=【)[ABCD]+(?=】)'
            answer_rule = r'[ABCD]+$'
        # 尝试匹配
        try:
            answer = re.findall(answer_rule, response)[0]
        except:
            # print(f'答案格式不匹配，尝试直接匹配第一次出现的字母...')
            # print(f"模型回复：{response}")
            try:
                if self.question_type == 'single':
                    answer_rule = r'\b[ABCD]'
                else:
                    # answer_rule = r'\b[ABCD]+'
                    answer_rule = r'[^ABCD]*([ABCD]+)[^ABCD]*'
                answer = re.findall(answer_rule, response)[0]
            except:
                # print('回答中没有包含答案！')
                answer = 'E'
        return answer
    
    def prompt_generation(self, question:str, options:str):
        def prefix_prompt_generation(question_type):
            if question_type == 'single':
                type = '单项选择题'
            elif question_type == 'multiple':
                type = '多项选择题'
            elif question_type == 'mix':
                type = '不定项选择题'
            else:
                type = '案例分析简答题'
            return f'你是一位专业的心理咨询助手，拥有丰富的心理学知识。现在有一道心理学知识的{type}，需要你利用自己的心理学知识进行解答\n'
        def question_prompt_generation(question_type, question, options=''):
            if question_type == 'single':
                prompt = '只有一个'
            elif question_type == 'multiple':
                prompt = '有多个'
            elif question_type == 'mix':
                prompt = '可能不止一个'
            else:
                prompt = '简答题'
            return f'选项中{prompt}是正确答案，请从ABCD四个选项中选出你认为正确的选项。【题目：{question}\n选项：{options}】\n'
        def answer_prompt_generation(question_type):
            if question_type == 'single':
                prompt = '【你认为正确的选项】'
                example = '【A】'
            elif question_type == 'multiple':
                prompt = '【你认为正确的选项】'
                example = '【AB】'
            elif question_type == 'mix':
                prompt = '【你认为正确的选项】'
                example = '【A】或【AB】'
            else:
                prompt = '【你的回复】'
            return f'请将你的答案按照{prompt}的格式输出，请一定要包含【】。样例：{example}，这只是一个样例，请不要参考里面的选项\n'
        prefix_prompt = prefix_prompt_generation(self.question_type)
        question_prompt = question_prompt_generation(self.question_type, question, options)
        other_template = f'请不要质疑题目的正确性。如果你认为题干表述不清晰，请结合自己的心理学知识，给出自己的答案。\n'
        answer_prompt = f'答案（仅回复相应的字母编号）：'

        return prefix_prompt + question_prompt + other_template + answer_prompt
    

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
            prefix_prompt = '**说明**：你是一位专业的心理咨询助手，拥有丰富的心理学知识和心理咨询经验。现在有一道心理咨询背景下的案例分析题，需要你利用自己的心理学知识以及心理咨询经验进行解答。\n'
            question_prompt = f"**求助者的个人信息如下**：{case['general_info']}\n" + f"**心理咨询内容如下**：{case['case_introduction']}\n" + f"请仔细阅读上述个人信息和心理咨询内容，并根据上述信息，站在一名专业心理咨询师的角度对下面的问题进行解答。**回答时请紧扣题目，不要偏题**。\n" \
            + f"**问题**：{case['qa_pair'][i][0]}\n"  + f"**选项**：{case['qa_pair'][i][2]}\n"
            answer_prompt = '答案（仅回复相应的字母编号）：'
            prompts.append(prefix_prompt + question_prompt + answer_prompt)
        return prompts
    
    def essay_prompt_generation(self, index:int):
        case = self.cases[index]
        prompts = []
        for i in range(len(case['qa_pair'])):
            prefix_prompt = '**说明**：你是一位专业的心理咨询助手，拥有丰富的心理学知识和心理咨询经验。现在有一道心理咨询背景下的案例分析题，需要你利用自己的心理学知识以及心理咨询经验进行解答。\n'
            question_prompt = f"**求助者的个人信息如下**：【{case['general_info']}】\n" + f"**心理咨询内容如下**：【{case['case_introduction']}】\n" + f"**请仔细阅读上述个人信息和心理咨询内容，并根据上述信息，站在一名专业心理咨询师的角度对下面的问题进行解答。回答时请紧扣题目，不要偏题**。\n" + f"**问题**：{case['qa_pair'][i][0]}\n"
            answer_prompt = '你的解答是（仅回复你对这个问题的回答）：'
            prompts.append(prefix_prompt + question_prompt + answer_prompt)
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
        if self.question_type != 'essay':
            # answer_rule = r'[ABCD]+(?=】)'
            answer_rule = r'[ABCD]+$'
            try:
                # answer = re.findall(answer_rule, response)
                answer = re.findall(answer_rule, response)[0]
            except:   
                try:
                    answer_rule = r'[^ABCD]*([ABCD]+)[^ABCD]*'
                    answer = re.findall(answer_rule, response)[0]
                except:
                    # print(f'答案格式不匹配，模型回答：{response}')
                    answer = 'E'
        else:
            answer_rule = r'(?<=:).+?(?=})'
            answer = re.findall(answer_rule, response)
            # if len(answer) == 0:
                # print(f'答案格式不匹配...')
        return answer

    
if __name__ == '__main__':
    answer_files = ['answer.json']
    question_type = 'essay'
    question_files = [os.path.dirname(os.path.abspath(__file__)) + '/questions/cases/' + '11-13 案例混合选择.json']
    question_handler = CaseQuestionHandler(question_files, answer_files, question_type)
    question_handler.choice_prompt_generation(0)
