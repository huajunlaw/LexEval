import json
import os
import argparse, sys

from loguru import logger
from requests import post, get


def read_json(input_file):
    data_list = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                ex = json.loads(line)
                data_list.append(ex)
            except Exception as e:
                logger.info(line)
    return data_list


def get_models(endpoint="http://127.0.0.1:11434", api_key="xxx"):
    resp = get(
        f"{endpoint}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=1000,
    )
    logger.info(resp.text)
    return resp.json()
    

def completion(
    messages: list[dict[str, str]],
    endpoint="http://127.0.0.1:11434",
    api_key="xxx",
    model_name=""
):
    req_json = {"messages": messages, "repetition_penalty": 1.5}
    if model_name:
        req_json['model'] = model_name 
    logger.info(req_json)
    resp = post(
        f"{endpoint}/v1/chat/completions",
        json=req_json,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=1000,
    )
    logger.info(resp.text)
    return resp.json()


def main(argv):
    """生成LawBench."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint", dest="endpoint",
                  help="endpoint: it should be a url ")
    parser.add_argument("-m", "--model", dest="model",
                  help="model: it should be a str ")
    parser.add_argument("-k", "--key", dest="api_key",
                  help="key: it should be a str")
    parser.add_argument("-s", "--shot", dest="shot",
                  help="shot: it should be a str")
    args = parser.parse_args(argv)
    logger.info(args)
    endpoint = args.endpoint
    api_key = args.api_key
    shot = args.shot or "zero_shot"
    model_name = args.model or "lawchat"
    data_path = "./data/"
    logger.info(data_path)
    prediction_path = "./model_output"
    data_files = os.listdir(data_path)
    out_path = os.path.join(prediction_path, shot, model_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for data_file in data_files:
        input_file = os.path.join(data_path, data_file)
        if not os.path.exists(input_file):
            logger.info(input_file)
            continue
        output_file = os.path.join(out_path, f"{model_name}_{data_file}l")
        if os.path.exists(output_file):
            continue
        data_list = read_json(input_file)
        outfile = open(output_file, 'w', encoding='utf8')
        for item in data_list:
            promopt = f"{item['instruction']}\n{item['input']}"
            messages = [{"role": "system", "content": "你是一个法官，旨在针对各种案件类型、审判程序和事实生成相应的法院裁决。你的回答不能含糊、有争议或者离题"},{"role": "user", "content": promopt}]
            if len(json.dumps(messages)) > 18192:
                logger.info(len(json.dumps(messages)))
            resp = completion(messages, endpoint=endpoint, api_key=api_key, model_name=model_name)
            try:
                prediction = resp['choices'][0]['message']["content"]
            except Exception as e:
                logger.info(e)
                continue
            save_dict = {
                "input": promopt,
                "output": prediction,
                "answer": item["answer"],
            }
            outline = json.dumps(save_dict,ensure_ascii=False)+'\n'
            outfile.write(outline)
        outfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
