import re
import json
import pandas
import os

"""
对话系统
"""

class DLSystem:
    def __init__(self):
        self.load()
        self.available_nodes = ['买衣服_node1']
        # self.available_nodes = ['买衣服_node1', "看电影_node1"]

    def load(self):
        self.all_slot_info = {}
        self.load_excel_templet("scenario/slot_fitting_templet.xlsx")
        self.all_node_info = {}
        self.load_scenario("scenario/scenario-买衣服.json")
        # self.load_scenario("scenario-看电影.json")

    def load_excel_templet(self, path):
        df = pandas.read_excel(path)
        for i in range(len(df)):
            slot = df['slot'][i]
            query = df['query'][i]
            values = df['values'][i]
            self.all_slot_info[slot] = [query, values]

    def load_scenario(self, path):
        scen_name = path.split("-")[1].split(".")[0]
        with open(path, encoding="utf8") as f:
            data = json.loads(f.read())
            for node_info in data:
                node_name = node_info["id"]
                if "childnode" in node_info:
                    node_info["childnode"] = [scen_name + "_" + name for name in node_info["childnode"]]
                node_name = scen_name + "_" + node_name
                self.all_node_info[node_name] = node_info

    def get_intent(self, memory):
        score = -1
        hit_node = None
        for node in self.available_nodes:
            intent_score = self.get_intent_score(memory, node)
            if intent_score > score:
                score = intent_score
                hit_node = node
        memory["score"] = score
        memory["hit_node"] = hit_node
        print("get_intent_memory", memory)
        return memory

    #节点意图打分
    def get_intent_score(self, memory, node):
        scores = []
        for intent in self.all_node_info[node]["intent"]:
            score = self.sentence_match(intent, memory["query"])
            scores.append(score)
        return max(scores)

    #文本匹配算法
    def sentence_match(self, string1, string2):
        return len(set(string1) & set(string2)) / len(set(string1)|set(string2))

    #槽位抽取
    def get_slot(self, memory):
        for slot in self.all_node_info[memory["hit_node"]].get("slot", []):
            pattern = self.all_slot_info[slot][1]
            match = re.search(pattern, memory["query"])
            if match is not None:
                memory[slot] = match.group()
        print("get_slot_memory", memory)
        return memory

    def nlu(self, memory):
        memory = self.get_intent(memory)
        memory = self.get_slot(memory)
        print("nlu_memory", memory)
        return memory

    def dst(self, memory):
        for slot in self.all_node_info[memory["hit_node"]].get("slot", []):
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        return memory

    def take_action(self, memory):
        return memory

    def pm(self, memory):
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            memory = self.take_action(memory)
            self.available_nodes = self.all_node_info[memory["hit_node"]].get("childnode")
        else:
            memory["policy"] = "ask"
            self.available_nodes = [memory["hit_node"]]
        return memory

    def nlg(self, memory):
        if memory["policy"] == "reply":
            response =  self.all_node_info[memory["hit_node"]].get("response")
            response = self.fill_templet(memory, response)
        else:
            slot = memory["require_slot"]
            response = self.all_slot_info[slot][0]
        memory["response"] = response
        return memory

    def fill_templet(self, memory, response):
        for slot in self.all_node_info[memory["hit_node"]].get("slot", []):
            response = re.sub(slot, memory[slot], response)
        return response

    def run(self, query, memory):
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.pm(memory)
        memory = self.nlg(memory)
        return memory

if __name__ == "__main__":
    dl = DLSystem()
    print(dl.all_slot_info)
    print(dl.all_node_info)
    memory = {}
    while True:
        query = input("用户输入需要的服务：")
        memory = dl.run(query, memory)
        print(memory)
        if memory["score"]>0:
            print("系统回答：", memory["response"])
        else:
            query = input("该服务暂不提供，请重新输入需要的服务：")
            memory = dl.run(query, memory)
        print()


