import nltk
import csv
nltk.download('framenet_v17')
from nltk.corpus import framenet as fn

# 定义艺术相关的关键词
art_keywords = {"art", "creativity", "music", "painting", "sculpture", "dance", "aesthetic"}

# 打开CSV文件并写入标题行
with open("art_sentences.csv", "w", newline='', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Frame", "Sentence", "Core Elements"])

    # 遍历所有的框架
    for frame in fn.frames():
        frame_name = frame.name
        frame_elements = [e.name for e in frame.FE.values() if e.coreType == 'Core']
        
        # 遍历框架的所有示例句子
        for s in fn.exemplars(frame=frame_name):
            sentence_text = s['text']
            
            # 判断句子是否包含艺术关键词
            if any(keyword in sentence_text.lower() for keyword in art_keywords):
                # 将句子、框架名称、核心元素写入CSV文件
                csvwriter.writerow([frame_name, sentence_text, "+".join(frame_elements)])

print("艺术主题相关的句子已成功收集并存储到 art_sentences.csv 文件中。")
