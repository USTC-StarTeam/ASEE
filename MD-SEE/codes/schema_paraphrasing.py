import json
import re
import requests
import os

def read_penultimate_line(filename, key):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()  # 读取所有行到列表中
            if len(lines) >= 1:  # 确保文件至少有1行
                penultimate_line = lines[-1].strip()  # 获取倒数第1行
                penultimate_dict = json.loads(penultimate_line)  # 将行解析为字典
                return penultimate_dict.get(key)  # 获取指定的键对应的值
            else:
                return "文件没有足够的行。"
    except FileNotFoundError:
        return "文件未找到。"
    except Exception as e:
        return f"读取文件时发生错误：{e}"

def data_preprocess(path_1, path_2):
    # 正则表达式替换属性名的单引号为双引号
    def fix_quotes(json_str):
        # 替换所有单引号为双引号，并处理可能存在的嵌套引号问题
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        return json_str

    # 辅助函数：检查字符串是否是有效的 JSON
    def is_valid_json(json_str, idx):
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}, index is {idx}")
            return False

    with open(path_1, 'r', encoding='utf-8') as f:
        json_list = []
        for line in f:
            # 将每一行解析为 JSON 对象并加入列表
            json_obj = json.loads(line.strip())
            json_list.append(json_obj)

    # 修正 arguments_w_descrip 的 JSON 格式
    for idx, json_obj in enumerate(json_list):
        temp = json_obj["arguments_w_descrip"]
        cleaned_string = temp.replace('```json', '').replace('```', '').strip()

        # 2. 去除换行符
        cleaned_string = cleaned_string.replace('\n', '')
        # cleaned_string = cleaned_string.replace('\"', "'")

        # 3. 将值中的单引号替换为双引号，确保 key 不受影响
        # 首先将单引号转换为双引号，并且确保字符串内部的双引号被转义
        cleaned_string = re.sub(r'"(.*?)\'(.*?)\'(.*?)"', r'"\1\"\2\"\3"', cleaned_string)
        cleaned_string = re.sub(r"\\'", "'", cleaned_string)
        cleaned_string = cleaned_string.replace('\\"', "'")

        # 检查修正后的 JSON 是否有效
        if is_valid_json(cleaned_string, idx):
            json_obj["arguments_w_descrip"] = json.loads(cleaned_string)  # 再解析为字典
        elif re.findall(r'\"(.*?)\": \"(.*?)(?<!\\)\"(?:,|\n})', cleaned_string, re.DOTALL):
            matches = re.findall(r'\"(.*?)\": \"(.*?)(?<!\\)\"(?:,|\n})', cleaned_string, re.DOTALL)
            json_obj["arguments_w_descrip"] = {key: value for key, value in matches}
        else:
            print(f"number -----{idx} -----  Skipping invalid JSON for object: {cleaned_string}")

    # 保存为新的 JSONL 文件
    out_path = 'res/schema_arguments_w_descrip_cleaned.json'

    # 将数据保存为 JSONL 格式
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in json_list:
            # 将字典转换为 JSON 字符串，并写入文件
            json_line = json.dumps(item, ensure_ascii=False)  # ensure_ascii=False 保持中文字符不被转义
            f.write(json_line + '\n')  # 每个 JSON 对象独占一行

    data_2 = []
    with open(path_2, 'r', encoding='utf-8') as file:
        for line in file:
            temp = json.loads(line.strip())
            data_2.append(temp)

    # 现在你可以像处理普通字典一样处理data了
    # print(data_2[0])
    print(len(data_2))

    name_list, query_list, arg_descrip_list, dataset_list, task_list, lang_list = [], [], [], [], [], []
    data_1 = json_list
    for i in range(len(data_1)):
        name_list.append(data_1[i]['name'])
        query_list.append(data_2[i]['queries'])
        arg_descrip_list.append(data_1[i]['arguments_w_descrip'])
        dataset_list.append(data_2[i]['dataset'])
        task_list.append(data_2[i]['task'])
        lang_list.append(data_2[i]['lang'])

    out_path = 'res/json_for_topic_descrip.json'
    with open(out_path, 'w', encoding='utf-8') as file:
        for i in range(len(name_list)):
            json_obj = {
            "name":name_list[i],
            'queries':query_list[i],
            'arguments':arg_descrip_list[i],
            'dataset':dataset_list[i],
            'task':task_list[i],
            'lang':lang_list[i]
            }
            json_string = json.dumps(json_obj, ensure_ascii=False)
            # 写入文件，每个元素占一行
            file.write(json_string + '\n')

    print("data preprocessing finished")

    return out_path


def main_3(path_1, path_2):
    api_key = "your api"  # 替换为你的实际 API 密钥
    base_url="your base url" 

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    path_in = data_preprocess(path_1, path_2)
    data_1 = []
    with open(path_in, 'r', encoding='utf-8') as file:
        # 逐行读取
        for line in file:
            # 去除行尾的换行符和空白字符
            line = line.strip()
            # 检查行是否为空
            if line:
                # 将非空行的JSON字符串解析为Python字典
                data = json.loads(line)
                # 处理每行的数据
                # print(data)
                data_1.append(data)

    # print(data_1[0])
    print(len(data_1))

    count_filename = "res/count_descrip.json"
    # 获取当前工作目录
    current_directory = os.getcwd()
    # 构建文件的完整路径
    count_file_path = os.path.join(current_directory, count_filename)
    # 检查文件是否存在
    if os.path.isfile(count_file_path):
        print(f"文件 {count_filename} 存在于当前目录。")
        start = read_penultimate_line(count_filename, "number")
        start = int(start)
        start += 1
        print(f"start number = {start}")
    else:
        print(f"文件 {count_filename} 不存在于当前目录。")
        start = 0
        print(f"start number = {start}")

    # for i in range(start, len(data_1)):
    for i in range(start, 1):
        name, query_list, the_argument = data_1[i]["name"], data_1[i]['queries'], data_1[i]['arguments']

        user_content = f"""
        You are a helpful assistant, and our task is to perform information extraction. Given a topic, along with a sample query list for the information extraction to be conducted, as well as the extracted information results corresponding to these query lists, presented in dictionary form with properties and descriptions.

        The topic is:
        {name}

        Sample query list for the information extraction to be conducted:
        {query_list}

        The extracted information format corresponding to these query lists, presented in dictionary form with properties and descriptions:
        {the_argument}

        Please generate a description of this information extraction task. This description will be used for a retrieval task, and we want it to perform better in information retrieval, making it easier to match with relevant queries while making the differences with irrelevant queries more distinct. If the topic is in Chinese, the final result will be in Chinese, and if the topic is in English, the final result will be in English. Please return only the description of this information extraction task and nothing else!
        """

        the_json = {
            "model": "gpt-4-turbo-2024-04-09",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content}
            ]
        }

        response = requests.post(url, json=the_json, headers=headers, verify=False)

        descrip = response.json()['choices'][0]['message']['content']

        print("descrip--------\n", descrip)
        res = {
            "name": name,
            "description": descrip,
            'arguments': the_argument,
            'dataset': data_1[i]['dataset'],
            'task': data_1[i]['task'],
            'lang': data_1[i]['lang']
        }
        with open("res/schema_pool_final.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

        with open("res/count_descrip.json", "a", encoding="utf-8") as f:
            f.write(json.dumps({"number": i}, ensure_ascii=False) + '\n')

    return "res/schema_pool_final.json"

if __name__ == "__main__":
    # path_1 = "schema_arguments_w_descrip_0909_cleaned.jsonl"
    # path_2 = "./data/schema_pool_w_queries.json"
    path_1 = "res/schema_arguments_w_descrip.json"
    path_2 = "res/schema_pool_w_queries.json"

    main_3(path_1, path_2)
