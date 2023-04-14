import os
# import numpy as np
import pandas as pd
# from datasets import DatasetDict
import time
from transformers import (BertTokenizer,
                          BertModelWithHeads,
BertModel
                          )
import torch

# https://stackoverflow.com/questions/73024608/merge-multiple-batchencoding-or-create-tensorflow-dataset-from-list-of-batchenco

def predict(premise, hypothesis):
    results = []
    st = 0
    encodes = tokenizer(premise, hypothesis[0], padding=True, return_tensors="pt")
    print(type(encodes))
    '''
    for answ in hypothesis:
        encoded = tokenizer(premise, answ, padding=True, return_tensors="pt")
        encodes["input_ids"] = torch.cat([encodes["input_ids"], encoded["input_ids"]], dim=0)
        encodes["token_type_ids"] = torch.cat([encodes["token_type_ids"], encoded["token_type_ids"]], dim=0)
        encodes["attention_mask"] = torch.cat([encodes["attention_mask"], encoded["attention_mask"]], dim=0)
        # if torch.cuda.is_available():
        #  encoded.to("cuda")
        t1 = time.time()
        print(encoded)
        print(type(encoded))
        print(encoded["input_ids"])
        print(type(encoded["input_ids"]))
        logits = model(**encoded)[0]
        pred_class = torch.argmax(logits).item()
        t2 = time.time() - t1
        st += t2
        print(t2)
        sgm = torch.sigmoid(logits)
        results.append({"class": pred_class, "sigmoid:": max(sgm[0]).item()})
    print(st)'''
    print(encodes)
    return results



# model_name = "cointegrated/rubert-tiny2"
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)
model = BertModelWithHeads.from_pretrained(model_name)



adapter_name = "nli_adapter"
# mode_name =  "nli-adapter-bert-base-e15"
mode_name = "checkpoint-120000"
# mode_name = "nli-adapter-rubert-tiny2-e5"
# adapter_name = "my_adapter"

adapter_path = os.path.join(os.getcwd(), "models", mode_name)
adapter_path2 = os.path.join(os.getcwd(), "models", mode_name, adapter_name)

model.load_adapter(adapter_path2)
model.set_active_adapters(adapter_name)



q = "срок сдачи декларации по налогу на прибыль за 2022 г., "
hypothesis = ["если мы не подаем уведомление о начисленных налогах, а платим платежными поручениями по КБК?",
                "Срок сдачи декларации по налогу на имущество за 2022 год: 27 марта 2023 года. ",
             "Срок сдачи изменился в связи с введением ЕНП.Куда сдавать: в ИФНС.Форма: форма декларации изменилась.",
              "Образец заполнения можно скачать ниже.Кто сдает: организации, у которых есть имущество, облагаемое налогом.",
              "Срок сдачи декларации по налогу на прибыль за 2022 год: 27 марта 2023 года. ",
              "Срок сдачи декларации по налогу на прибыль за 2021 год: 30 марта 2022 года. "]


queries = [q] * len(hypothesis)
queries_hypothesis = [(q, h) for q, h in zip(queries, hypothesis)]
encodes = tokenizer.batch_encode_plus(queries_hypothesis)
print(type(encodes))
# model.estimate_tokens(**encodes)
logits = model(**encodes)

t = time.time()
r = predict(q, hypothesis)
print(time.time() - t)
print("r:\n", r)
