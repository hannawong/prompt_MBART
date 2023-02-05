import json
from seqeval.metrics import precision_score, recall_score, f1_score

reference = open("./text2text/translate/test_all.output").read().split("\n")[:-1]
reference_full = open("./text2text/test_all.output_full").read().split("\n")[:-1]
generated = json.load(open("./outputs/mbart_translate/translate_epoch8result.txt","r"))

dic = { "en":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0, "intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[], "slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "fr":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "de":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "es":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "zh":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0}}

for i in range(len(reference)):
    slot_ground_truth = [] ## the slot of full sentence
    ref_full = reference_full[i].split("<intent-")[0].strip().lower()
    ref_lang = reference_full[i].split("lang-")[1].split(">")[0]
    gen = generated[i].split("<s>")[0]
    gen = gen.split("@")[1].strip()
    word2idx = {}
    cnt = 0
    for item in ref_full.split(">"):
        if item == "": continue
        word = item.split("<")[0].strip()
        if word not in word2idx:
            word2idx[word] = [cnt]
        else:
            word2idx[word].append(cnt)
        slot_label = item.split("<")[1].strip()
        slot_ground_truth.append(slot_label)
        cnt += 1

    gen_slot_labels = ['o'] * cnt
    for item in gen.split(">"):
        if item == "": continue
        word = item.split("<")[0].strip()
        slot_label = item.split("<")[1].strip() if "<" in item else "O"
        if word in word2idx and len(word2idx[word]):
            gen_slot_labels[word2idx[word][0]] = slot_label
            word2idx[word] = word2idx[word][1:]
    for i in range(len(slot_ground_truth)):
        slot_ground_truth[i] = slot_ground_truth[i][0].upper() + slot_ground_truth[i][1:]
    for i in range(len(gen_slot_labels)):
        gen_slot_labels[i] = gen_slot_labels[i][0].upper() + gen_slot_labels[i][1:]
    #assert len(slot_ground_truth) == len(gen_slot_labels)
    dic[ref_lang]["pred_list"].append(gen_slot_labels)
    dic[ref_lang]["ground_truth_list"].append(slot_ground_truth)

tot = 0

for lang in ["en","de","es","fr","zh"]:
    dic[lang]["slot_f1"] = f1_score(dic[lang]["pred_list"], dic[lang]["ground_truth_list"])
    print(lang,":",dic[lang]["slot_f1"])
    tot += dic[lang]["slot_f1"]
print(tot/5)