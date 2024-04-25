# Evaluation

Paper link.

## Usage Guide

For running a supported model on this benchmark, please run the following command:
```python
python main.py \
    --model_name [model_name] \
    --knowledge_type [theory|moral|cases] \
    --question_type [single|multiple|mix|essay] \
    --scope [second|third|all] \
    --sleep [sleep_time] \
    --local [True|False] \
    --all_in_one
```
`--model_name`: the name of model to evaluate. Default to "qwen1.5-7b-chat" (use Qwen API | local).

`--knowledge_type`: the examination knowledge points of the questions. Default to "theory".

`--question_type`: the answer format of the questions. Default to "single". Please note that [**theory**, **moral** --> **single** or **multiple**], [**cases** --> **mix** or **essay**].

`--scope`: the level range of questions. Default to "all".

`--sleep`: the sleep time after completing each question. It is mainly to deal with the QPS limit when calling the API. Default to "0.1".

`--local`: the indicator that identifies whether the model to evaluate is locally deployed. Default to "False".

`--all_in_one`: Complete the experiment of all types of questions at once. Default to "False".

> **Notice:** The model has not been uploaded. If you want to evaluate the locally deployed model (e.g. qwen1.5-7b-chat), you need to download the corresponding model file from [huggingface](https://huggingface.co/) and place it in the *models* folder in the root directory. And if you want to use the API to access LLMs, you need to apply for an API-key from the official website of [Qwen](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-7b-14b-72b-api-detailes) or [Qianfan](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu) to call the API.

## Supported Models
1. qwen1.5-7b-chat (**Qwen API** | **local**)
2. qwen1.5-14b-chat (**Qwen API** | **local**)
3. baichuan2-7b-chat (**local**)
4. baichuan2-13b-chat(**local**)
5. chatglm3-6b-32k (**local**)
6. qianfan-chinese-llama-2-7b (**Qianfan API**)
7. qianfan-chinese-llama-2-13b (**Qianfan API**)
8. chinese-alpaca-2-7b (**local**)
9. chinese-alpaca-2-13b (**local**)
10. yi-6b-chat (**local**)
11. yi-34b-chat (**Qianfan API**)
12. ... (To be continued)

## Citation
```
```

