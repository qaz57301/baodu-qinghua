import torch
import torch.nn as nn

from nnlm import LanguageModelWithBert
from lm import LanguageModel, get_vocab, hidden_size, window_size



def load_model():
    bert_path = "C:\\yuanzhouli\\AllCodeRelated\\bert\\bert-base-chinese"
    bert_model = LanguageModelWithBert(bert_path)
    bert_model.load_state_dict(torch.load("nn_model.pth"))
    lstm_model = LanguageModel(get_vocab("vocab2.txt"), hidden_size)
    lstm_model.load_state_dict(torch.load("model.pth"))
    return bert_model, lstm_model


def get_tongyin_word_dict(tongyin_path):
    tongyin_word_dict = {}
    with open(tongyin_path, "r", encoding="utf-8") as f:
        lines = [e.strip().split(" ") for e in f.readlines()]
        for line in lines:
            tongyin_word_dict[line[0]] = line[1]
    return tongyin_word_dict


def test(test_strings, tongyin_path):
    bert_model, lstm_model = load_model()
    tongyin_word_dict = get_tongyin_word_dict(tongyin_path)
    for test_string in test_strings:
        bert_ppl = bert_model.cal_ppl(test_string, window_size)
        lstm_ppl = lstm_model.cal_ppl(test_string, window_size)

        bert_string = ""
        lstm_string = ""
        for i in range(len(test_string)):
            word = test_string[i]
            tongyin_words = tongyin_word_dict.get(word)
            bert_replace_word = ""
            lstm_replace_word = ""
            if tongyin_words:
                ori_string = list(test_string)

                bert_min_ppl = bert_ppl
                lstm_min_ppl = lstm_ppl
                for tongyin_word in tongyin_words:
                    ori_string[i] = tongyin_word
                    new_bert_ppl = bert_model.cal_ppl("".join(ori_string), window_size)
                    new_lstm_ppl = lstm_model.cal_ppl("".join(ori_string), window_size)
                    if new_bert_ppl < bert_min_ppl:
                        bert_min_ppl = new_bert_ppl
                        bert_replace_word = tongyin_word
                    if new_lstm_ppl < lstm_min_ppl:
                        lstm_min_ppl = new_lstm_ppl
                        lstm_replace_word = tongyin_word
            bert_string += bert_replace_word if bert_replace_word != "" else word
            lstm_string += lstm_replace_word if lstm_replace_word != "" else word
        if bert_string != test_string:
            final_ppl = bert_model.cal_ppl(bert_string, window_size)
            print("bert模型预测将：%s 替换成 %s，ppl从%f变成%f" % (test_string, bert_string, bert_ppl, final_ppl))
        else:
            print("bert模型预测无需纠错")
        if lstm_string != test_string:
            final_ppl = lstm_model.cal_ppl(lstm_string, window_size)
            print("lstm模型预测将：%s 替换成 %s，ppl从%f变成%f" % (test_string, lstm_string, lstm_ppl, final_ppl))
        else:
            print("lstm模型预测无需纠错")


if __name__ == '__main__':
    test_strings = ["今天田气真豪", "与会交易所老总俊表示", "基础制毒建设的质变"]
    tongyin_path = "tongyin.txt"
    test(test_strings, tongyin_path)