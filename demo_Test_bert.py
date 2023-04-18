import json
import torch
from transformers import AutoTokenizer
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
from transformers import HfArgumentParser, TrainingArguments, Trainer, set_seed
from datasets import load_dataset, Dataset
from loguru import logger

tokenizer = BertTokenizer.from_pretrained(r"C:\Users\49344\Desktop\AI\bert1")
model = BartForConditionalGeneration.from_pretrained(r"C:\Users\49344\Desktop\AI\bert1")

class CscDataset(object):
    def __init__(self, file_path):
        self.data = json.load(open(file_path, 'r', encoding='utf-8'))

    def load(self):
        data_list = []
        for item in self.data:
            data_list.append(item['original_text'] + '\t' + item['correct_text'])
            if len(data_list) > 10000:
                break
        return {'text': data_list}


def bart_correct(tokenizer, model, text: str, max_length: int = 128):
    import numpy as np
    inputs = tokenizer.encode(text, padding=True, max_length=32, truncation=True,
                              return_tensors='pt')
    model.eval()
    with torch.no_grad():
        res = model(inputs).logits
        res = np.argmax(res[0], axis=1)
        res = res[1:-1]
        decode_tokens = tokenizer.decode(res, skip_special_tokens=True).replace(' ', '')
    return decode_tokens

def tokenize_dataset(tokenizer, dataset, max_len):
    def convert_to_features(example_batch):
        src_texts = []
        trg_texts = []
        for example in example_batch['text']:
            terms = example.split('\t', 1)
            src_texts.append(terms[0])
            trg_texts.append(terms[1])
        input_encodings = tokenizer.batch_encode_plus(
            src_texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
        )
        target_encodings = tokenizer.batch_encode_plus(
            trg_texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
        )

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'target_ids': target_encodings['input_ids'],
            'target_attention_mask': target_encodings['attention_mask']
        }

        return encodings

    dataset = dataset.map(convert_to_features, batched=True)
    # Set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
    dataset.with_format(type='torch', columns=columns)
    # Rename columns to the names that the forward method of the selected
    # model expects
    dataset = dataset.rename_column('target_ids', 'labels')
    dataset = dataset.rename_column('target_attention_mask', 'decoder_attention_mask')
    dataset = dataset.remove_columns(['text'])
    return dataset

d = CscDataset("train.json")
data_dict = d.load()
train_dataset = Dataset.from_dict(data_dict, split='train')

d = CscDataset("test.json")
data_dict = d.load()
valid_dataset = Dataset.from_dict(data_dict, split='test')

logger.info(train_dataset)
logger.info(valid_dataset)

train_data = tokenize_dataset(tokenizer, train_dataset, 128)
valid_data = tokenize_dataset(tokenizer, valid_dataset, 128)

training_args = TrainingArguments(
    output_dir='./results',         # output directory 结果输出地址
    num_train_epochs=1,          # total # of training epochs 训练总批次
    per_device_train_batch_size=32,  # batch size per device during training 训练批大小
    per_device_eval_batch_size=32,   # batch size for evaluation 评估批大小
    logging_dir='./logs/rn_log',    # directory for storing logs 日志存储位置
    learning_rate=1e-4,             # 学习率
    save_steps=False,# 不保存检查点
    logging_steps=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
)
# trainer.train()
# ##模型保存
# model.save_pretrained("result_bart/")

new_model = BartForConditionalGeneration.from_pretrained("./result_bart/")
text2text_generator = Text2TextGenerationPipeline(new_model, tokenizer)
print(text2text_generator("中国的首都是[MASK]京", max_length=50, do_sample=False))
print(text2text_generator("中国的[MASK]都是北京", max_length=50, do_sample=False))
print(text2text_generator("中国的首都[MASK]北京", max_length=50, do_sample=False))
print(text2text_generator("中[MASK]的首都是北京", max_length=50, do_sample=False))
print(text2text_generator("中过的首都是北京", max_length=50, do_sample=False))
print(text2text_generator("中国的首都你北京", max_length=50, do_sample=False))
print(text2text_generator("你在他相还好吗", max_length=50, do_sample=False))
print(text2text_generator("好好学习[MASK]天向上", max_length=50, do_sample=False))
print(text2text_generator("好好学习你天向上", max_length=50, do_sample=False))
print(text2text_generator("一定要把民族汽车贫牌搞上去", max_length=50, do_sample=False))
print(text2text_generator("一定要把眀族汽车玶牌搞上去", max_length=50, do_sample=False))
print(text2text_generator("检查风扇是否正常云转", max_length=50, do_sample=False))
print(bart_correct(tokenizer, new_model,"中国的首都是杯京",32))
