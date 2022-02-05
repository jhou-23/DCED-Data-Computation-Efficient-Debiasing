from datasets import load_dataset, Dataset
from transformers import AutoModelForMaskedLM


model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
model.save_pretrained("./bert-test-uncased/", save_config=True)


# dataset = load_dataset("wikipedia", '20200501.en')
# dataset.cleanup_cache_files()
# print(len(dataset["train"]))

# word_map = {"african":"caucasian", "African":"Caucasian", "caucasian":"african", "Caucasian":"African", "black":"white", "Black":"White", "white":"black", "White":"Black", "africa":"europe", "Africa":"Europe", "europe":"africa", "Europe":"Africa"}

# def onein(a):
#     for word in word_map.keys():
#         if word in a["text"]:
#             return True
#     return False

# dataset["train"] = dataset["train"].filter(lambda example: onein(example))
# print(len(dataset["train"]))
# print(dataset["train"][0])



# dataset = Dataset.from_file("/root/.cache/huggingface/datasets/wikipedia/20200501.en/1.0.0/009f923d9b6dd00c00c8cdc7f408f2b47f45dd4f5fb7982a21f9448f4afbe475/cache-bfef86996dc3563c.arrow")
# print(len(dataset))
# print(dataset)















# def replace_tgt(example):
#     words = example["text"].split(" ")

#     text = ""
#     for word in words:
#         if word.strip() in word_map.keys():
#             if word.strip() == word:
#                 text +=  word_map[word] + " "
#             else:
#                 text += word_map[word.strip()] + "\n "
#         else:
#             text += word + " "
    
#     example["text"] = text
#     return example

# # index = -1
# # for i in range(len(dataset["train"])):
# #     if "block" in dataset["train"][i]["text"]:
# #         print(dataset["train"][i]["text"])
# #         index = i
# #         break

# # dataset = dataset.map(replace_tgt)

# # print(dataset["train"][index]["text"])

# a = dict(zip(word_map.keys(), [0] * len(word_map)))

# total = 0
# flag = True
# for i in range(len(dataset["train"])):
#     for key in word_map.keys():
#         if key in dataset["train"][i]["text"]:
#             a[key] += 1
#             if flag:
#                 total += 1
#                 flag = False
#     flag = True

# print(total)
# print(a)
"""
6078422
1382313
{'african': 3142, 'African': 221210, 'caucasian': 1019, 'Caucasian': 6991, 'black': 261907, 'Black': 283048, 'white': 268656, 'White': 264603, 'africa': 4762, 'Africa': 360522, 'europe': 2957, 'Europe': 547185}
"""