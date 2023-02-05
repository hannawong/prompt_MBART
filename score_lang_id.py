import json 
reference = open("./text2text/language_identification/test_all.output").read().split("\n")[:-1]
generated = json.load(open("./outputs/mbart_langid_more/epoch2result.txt","r"))
dic = { "en":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0, "intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[], "slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "fr":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "de":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "es":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "zh":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0}}

for i in range(len(reference)):
    ref = reference[i].split("<s>")[0].strip().lower()
    gen = generated[i].split("<s>")[0].strip().lower()
    lang_ref = ref.split("@")[1].strip()
    lang_gen = gen.split("@")[1].strip()
    print(lang_gen,lang_ref)
    if lang_ref == lang_gen:
        dic[lang_ref]["lang_correct"] += 1
    dic[lang_ref]["lang_tot"] += 1
print(dic)
