import json 
reference = open("./text2text/intent_prediction/test_all.output").read().split("\n")[:-1]
reference_lang = open("./text2text/language_identification/test_all.output").read().split("\n")[:-1]
generated = json.load(open("./outputs/mbart_intent_langspecific_vertical/epoch9result.txt","r"))
dic = { "en":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0, "intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[], "slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "fr":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "de":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "es":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "zh":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0}}

intent_correct = 0
intent_tot = 0
for i in range(len(reference)):
    ref = reference[i].split("<s>")[0].strip().lower()
    gen = generated[i].split("<s>")[0].strip().lower()
    ref_lang = reference_lang[i].split("@")[1].strip().lower()
    intent_ref = ref.split("@")[1].strip()
    intent_gen = gen.split("@")[1].strip()
    print(intent_gen,intent_ref)
    if intent_ref == intent_gen:
        dic[ref_lang]["intent_correct"] += 1
    dic[ref_lang]["intent_tot"] += 1
tot = 0
for key in dic.keys():
    print(key,":",dic[key]["intent_correct"] / dic[key]["intent_tot"])
    tot += dic[key]["intent_correct"] / dic[key]["intent_tot"]
print(tot/5)

