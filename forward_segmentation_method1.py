# 分词方法：最大正向切分的第一种实现方式

import re
import time

#加载词典
def load_word_dict(path):
    max_word_length=0
    word_dict={}                   # 用set 也是可以的。用list会很慢
    with open(path,encoding='utf8') as f:
        for line in f:
            word=line.split()[0]
            word_dict[word]=0
            max_word_length=max(max_word_length,len(word))
            # print(max_word_length)
    return word_dict, max_word_length

#先确定最大词长度
#从长向短查找是否有匹配的词
#找到后移动窗口


def cut_method1(string, word_dict, max_len):
    words=[]
    # string!='' string不为空就执行
    while string!='':
        lens = min(max_len, len(string))
        print("lens", lens)
        word = string[:lens]
        print(word)
        #如果word不在dict里面，则往前递减，直到出现在dict里面
        while word not in word_dict:
            if len(word)==1:
                break
            word=word[:len(word)-1]
            print(word)
        words.append(word)
        #如果word在dict里面 string就是去掉word剩下的
        string=string[len(word):]
    #string最终为空
        # print("string",string)
    return words

#cut_method是切割函数
#output_path是输出路径
def main(cut_method, input_path, output_path):
    #读取字典表和最长字符串的长度
    word_dict, max_word_length = load_word_dict("dict.txt")
    # 将分词结果写入文件
    writer = open(output_path, "w", encoding="utf8")
    start_time = time.time()
    with open(input_path,encoding="utf8") as f:
        #对input_path 文件中的每行进行分词
        for line in f:
            words = cut_method(line.strip(), word_dict, max_word_length)
            writer.write("/".join(words)+"\n")
    writer.close()
    print("耗时：", time.time()-start_time)
    return



path="dict.txt"
# string="测试字符串湖南人"
word_dict, max_len = load_word_dict(path)
# print(word_dict)
# print(max_len)
# words = cut_method1(string, word_dict, max_len)
# print("words", words)
main(cut_method1, "corpus.txt", "cut_method1_output.txt")