from QuestionHandler import GeneralQuestionHandler, CaseQuestionHandler
from LLMTransfer import LLMTransfer
from SimilarMetric import compute_metrics

import os
import time
import pandas as pd
import datetime
import argparse
from alive_progress import alive_bar

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="qwen1.5-7b-chat") # 模型代号
parser.add_argument("-k", "--knowledge_type", type=str, default="theory") # 题目考察的知识类型，可选值：theory, moral, cases
parser.add_argument("-q", "--question_type", type=str, default="single") # 题目类型，可选值：single, multiple, mix, essay（只有案例题有mix和essay）
parser.add_argument("-l", "--local", type=bool, default=False) # 是否为本地部署模型
args = parser.parse_args()

model_name = args.model_name
knowledge_type = args.knowledge_type
question_type = args.question_type
b_local = args.local


# model_name = 'qwen1.5-14b-chat'
# model_name = 'baichuan2-13b-chat'
# model_name = 'chinese-alpaca-2-13b'
# model_name = 'sft_qwen1.5_4k'
# model_name = "chatglm3-6b-32k"
# knowledge_type = 'theory'
# question_type = 'multiple'

# 题目文件名
question_file_dict = {}
question_file_dict['theory_single_question_files'] = ['11-13,17-18 理论单选.json', '14-16 理论单选.json']
question_file_dict['theory_multiple_question_files'] = ['11-16 理论多选.json', '17-18 理论多选.json']
question_file_dict['moral_single_question_files'] = ['14-16 职业道德单选.json']
question_file_dict['moral_multiple_question_files'] = ['11-18 职业道德多选.json', '17-18 职业道德多选.json']
question_file_dict['cases_mix_question_files'] = ['11-13 案例混合选择.json', '14-16 案例混合选择.json', '17-18 案例混合选择.json']
question_file_dict['cases_essay_question_files'] = ['11-13 案例问答.json', '14-15 案例问答.json', '17-18 案例问答.json']
question_file_name = question_file_dict[f'{knowledge_type}_{question_type}_question_files']


'''
answer_type: 0：完全正确，1：少选，2：多选或错选
'''
def correct_rate_compute(model_answer:str, true_answer:str) -> tuple:
    if model_answer == true_answer:
        return (0, 1)
    else:
        model_answer_set = set(model_answer)
        true_answer_set = set(true_answer)
        if model_answer_set.issubset(true_answer_set):
            return (1, float(len(model_answer_set)) / len(true_answer_set))
    return (2, 0)
    

'''
负责处理非案例问题（一题一问）的函数
'''
def general_question_run(model_name:str, question_files:str, result_file:str, question_type:str) -> None:
    # Handler的作用包括：问题提取，答案提取
    question_handler = GeneralQuestionHandler(question_files, result_file, question_type)
    # 模型调用，包括API调用与本地调用
    model_api = LLMTransfer(model_name, b_local=b_local)
    # 记录评估结果的参数
    sum_correct_rate, completely_correct_count, partially_correct_count, wrong_count, total_count = 0., 0, 0, 0, len(question_handler.questions)
    answers = pd.DataFrame({'模型答案': [], '真实答案': []})
    # 开始测评
    print('-' * 50 + "测评开始" + '-' * 50)
    print(f"模型代号：{model_name}，测评知识类型：{knowledge_type}，题目类型：{question_type}，题目数量：{total_count}。")
    with alive_bar(total_count, title=f"正在测评{model_name}模型", bar="smooth", spinner="waves2") as bar:
        for i in range(total_count):
            # 生成prompt
            prompt = question_handler.prompt_generation(question_handler.questions[i], question_handler.contents[i]['options'])
            # 调用接口
            response = model_api.call_with_prompt(prompt)
            # 提取答案
            true_answer = question_handler.true_answers[i]
            model_answer = question_handler.model_answer_processing(response)
            # 计算完全正确、部分正确、多选与错选的题目个数
            answer_type, correct_rate = correct_rate_compute(model_answer, true_answer)
            if answer_type == 0:
                completely_correct_count += 1
            elif answer_type == 1:
                partially_correct_count += 1
            else:
                wrong_count += 1
            sum_correct_rate += correct_rate
            # 结果汇总
            answers.loc[i] = [model_answer, true_answer]
            # print(f'{i+1}. 模型答案：{model_answer} 正确答案：{true_answer}')
            # 输出进度
            if (i + 1) % 50 == 0:
                if question_type == 'single':
                    print(f'目前已回答{i + 1}题，{completely_correct_count}题正确，{wrong_count}题错误。正确率：{(completely_correct_count / (i + 1)) * 100}%')
                else:
                    print(f'目前已回答{i + 1}题，{completely_correct_count}题完全正确，{partially_correct_count}题部分正确，{wrong_count}题多选或错选。严格正确率：{(completely_correct_count / (i + 1)) * 100}%，弹性正确率：{(sum_correct_rate / (i + 1)) * 100}%')
            time.sleep(0.1)
            bar()
    
    # 测评总结
    if question_type == 'single':
        summary = f'共{total_count}题，{completely_correct_count}题正确，{wrong_count}题错误。正确率：{(completely_correct_count / (i + 1)) * 100}%'
    else:
        summary = f'共{total_count}题，{completely_correct_count}题完全正确，{partially_correct_count}题部分正确，{wrong_count}题多选或错选。严格正确率：{(completely_correct_count / total_count) * 100}%，弹性正确率：{(sum_correct_rate / total_count) * 100}%'
    print(summary)

    # 保存结果
    answers.loc[len(answers.index)] = ['测评最终结果', summary]
    answers.index = range(1, len(answers) + 1)
    answers.to_csv(result_file)
    print("测评结果已保存至：" + result_file)
    print('-' * 50 + "测评结束" + '-' * 50)


'''
负责处理案例问题（一题多问）的函数
'''
def case_question_run(model_name:str, question_files:str, result_file:str, question_type:str) -> None:
    # Handler的作用包括：问题提取，答案提取
    question_handler = CaseQuestionHandler(question_files, result_file, question_type)
    # 模型调用，包括API调用与本地调用
    model_api = LLMTransfer(model_name, b_local=b_local)
    # 记录评估结果的参数
    sum_correct_rate, completely_correct_count, partially_correct_count, wrong_count, total_count, missing_count = 0., 0, 0, 0, 0, 0
    questions_len = len(question_handler.contents)
    answers = pd.DataFrame({'题号': [], '模型答案': [], '正确答案': []})
    # 开始测评
    print('-' * 50 + "测评开始" + '-' * 50)
    print(f'模型代号：{model_name}，测评知识类型：{knowledge_type}，题目类型：{question_type}，大题数量：{questions_len}。')
    with alive_bar(questions_len, title=f"正在测评{model_name}模型", bar="smooth", spinner="waves2") as bar:
        for i in range(questions_len):
            prompts = question_handler.prompt_generation(i)
            for j, prompt in enumerate(prompts):
                # 调用接口，提交问题
                response = model_api.call_with_prompt(prompt)
                # 从回复中提取答案
                true_answer = question_handler.true_answers[i][j]
                if question_type == 'mix':
                    # 选择题需要正确地提取选项
                    model_answer = question_handler.model_answer_processing(response)
                    answer_type, correct_rate = correct_rate_compute(model_answer, true_answer)
                    if model_answer == 'E':
                        missing_count += 1
                    elif answer_type == 0:
                        completely_correct_count += 1
                    elif answer_type == 1:
                        partially_correct_count += 1
                    else:
                        wrong_count += 1
                    sum_correct_rate += correct_rate
                else:
                    # 对于简答题，整个回复都是模型的答案
                    model_answer = response
                # 结果汇总
                answers.loc[total_count] = [f"{i+1}.{j+1}", model_answer, true_answer]
                total_count += 1
                # print(f'{i+1}.{j+1}. 模型答案：{model_answer} 正确答案：{true_answer}')
            # 输出进度
            if question_type == "mix" and (i + 1) % 10 == 0:
                print(f'目前已回答{i + 1}道大题，共包含{total_count}道具体题目。其中，{completely_correct_count}题完全正确，{partially_correct_count}题部分正确，{wrong_count}题多选或错选，{missing_count}题未答或漏答。严格正确率：{(completely_correct_count / total_count) * 100}%，弹性正确率：{(sum_correct_rate / total_count) * 100}%')
            time.sleep(0.1)
            bar()
    
    if question_type == "mix":
        summary = f'共{questions_len}道大题，{total_count}道具体题目。其中，{completely_correct_count}题完全正确，{partially_correct_count}题部分正确，{wrong_count}题多选或错选，{missing_count}题未答或漏答。严格正确率：{(completely_correct_count / total_count) * 100}%，弹性正确率：{(sum_correct_rate / total_count) * 100}%'
    else:
        answer_pairs = [answers["模型答案"].values.tolist(), answers["正确答案"].values.tolist()]
        result = compute_metrics(answer_pairs)
        summary = f'共{questions_len}道大题，{total_count}道具体题目。ROUGE与BLEU得分：{result}\n'
    print(summary)

    # 保存结果
    answers.loc[len(answers.index)] = ['', '测评最终结果', summary]
    answers.index = range(1, len(answers) + 1)
    answers.to_csv(result_file)
    print("测评结果已保存至：" + result_file)
    print('-' * 50 + "测评结束" + '-' * 50)


if __name__ == '__main__':
    question_files = [os.path.dirname(os.path.abspath(__file__)) + '/questions/' + knowledge_type + '/' + file for file in question_file_name]
    result_file = model_name + '_' + datetime.datetime.now().strftime('%Y%m%d-%Hh%Mm%Ss') + '.csv'  # 测评结果保存位置
    result_file = os.path.dirname(os.path.abspath(__file__)) + '/results/' + knowledge_type + '/' + question_type + '/' + result_file
    # 根据题目类型，运行对应的测评函数
    if knowledge_type == "theory" or knowledge_type == "moral":
        general_question_run(model_name, question_files, result_file, question_type)
    else:
        case_question_run(model_name, question_files, result_file, question_type)
