import json
import csv


def parse_to_dict_list(data):
    index = 0
    final_list = []
    for title in data:
        title_name = title['title']
        for para in title['paragraphs']:
            context = para['context']
            for qas in para['qas']:
                answers = set()
                if qas['is_impossible']:
                    answers.add('')
                question = qas['question']
                for ans in qas['answers']:
                    answers.add(ans['text'])
                for answer in answers:
                    entry = {'id': index, 'title': title_name, 'context': context, 'question': question,
                             'answer': answer}
                    final_list.append(entry)
                    index = index + 1
    return final_list


if __name__ == "__main__":

    # Parse squad json to csv
    with open('./squad_dataset/dev-v2.0.json') as f:
        dev_data = json.load(f)
    dev_data = dev_data['data']

    print("Types of paragraphs in Squad Dataset:\n")
    for i in dev_data:
        print(i['title'])

    dev_data_dict_list = parse_to_dict_list(dev_data)

    try:
        with open('./squad_dataset/dev-v2.0.csv', 'w+', encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dev_data_dict_list[0].keys())
            writer.writeheader()
            for entry in dev_data_dict_list:
                writer.writerow(entry)
    except IOError as e:
        print(e)

    with open('./squad_dataset/train-v2.0.json') as f:
        train_data = json.load(f)
    train_data = train_data['data']

    train_data_dict_list = parse_to_dict_list(train_data)

    try:
        with open('./squad_dataset/train-v2.0.csv', 'w+', encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=train_data_dict_list[0].keys())
            writer.writeheader()
            for entry in train_data_dict_list:
                writer.writerow(entry)
    except IOError as e:
        print(e)
