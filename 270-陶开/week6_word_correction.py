import json
import copy
from ngram_language_model import NgramLanguageModel
"""
文本纠错demo
加载同音字字典
加载语言模型
基本原理：
对于文本中每一个字，判断在其同音字中是否有其他字，在替换掉该字时，能使得语言模型计算的成句概率提高
"""


class Corrector:
    def __init__(self, language_model):
        #语言模型
        self.language_model = language_model
        #候选字字典
        self.sub_dict = self.load_tongyinzi("tongyin.txt")
        #成句概率的提升超过阈值则保留修改
        self.threshold = 7

    #实际上不光是同音字，同形字等也可以加入，本质上是常用的错字
    def load_tongyinzi(self, path):
        tongyinzi_dict = {}
        with open(path, encoding="utf8") as f:
            for line in f:
                char, tongyin_chars = line.split()
                tongyinzi_dict[char] = list(tongyin_chars)
        return tongyinzi_dict

    def prob_string(self, sentence):
        prob = self.language_model.calc_sentence_ppl(sentence)
        return prob

    #纠错逻辑
    def correction(self, string):
        # todo  自己尝试实现
        tongyinzi_dict = self.sub_dict
        # print(prob)
        n = len(string)
        print(n)
        if n==1:
            max_prob = self.language_model.calc_sentence_ppl(string)
            for word in tongyinzi_dict[string]:
                new_string_prob = self.language_model.calc_sentence_ppl(string.replace(string[n-1], word))
                if new_string_prob>max_prob:
                    max_prob=new_string_prob
                    string=string.replace(string[n-1], word)
                    return max_prob,string
        return self.correction(string[n-2]+string[n-1])








                



corpus = open("财经.txt", encoding="utf8").readlines()
lm = NgramLanguageModel(corpus, 3)
# print("每国货币政册空间不大",lm.calc_sentence_ppl("每国货币政册空间不大"))
# print("美国货币政策空间不大",lm.calc_sentence_ppl("美国货币政策空间不大"))
# print("镁国货币政策空间不大",lm.calc_sentence_ppl("镁国货币政策空间不大"),lm.ngram_count_dict)


language_model = NgramLanguageModel(corpus,3)
cr = Corrector(language_model)
# print(cr.sub_dict)
# print(cr.language_model)
string = "每国货币政册空间不大"  #美国货币政策空间不大
fix_string = cr.correction(string)
# print(fix_string)
prob=cr.prob_string(sentence=string)
# print(prob)

# print("修改前：", string)
# print("修改后：", fix_string)

# wordlist = list("每国货币政册空间不大")
# print(wordlist)
# p0 = lm.calc_sentence_ppl("每国货币政册空间不大")
# tongyinzi = list("镁美媒没酶味枚某霉妹梅沫煤墨眉玫")
# for word in tongyinzi:
#     sentence = word+""+"".join(wordlist[1:])
#     print(sentence)
#     prob_word_sentence = lm.calc_sentence_ppl(sentence)
#     print(prob_word_sentence)
