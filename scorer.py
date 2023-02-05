import json
from seqeval.metrics import precision_score, recall_score, f1_score

reference = open("./text2text/test_all.output").read().split("\n")[:-1]
generated = json.load(open("./outputs/mbart/result.txt","r"))
reference_full = open("./text2text/test_all.output_full").read().split("\n")[:-1]
dic = { "en":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0, "intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[], "slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "fr":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "de":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "es":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0},\
        "zh":{"lang_correct":0, "lang_tot":0,"lang_acc" : 0.0,"intent_correct" : 0, "intent_tot":0, "intent_acc":0,"pred_list":[], "ground_truth_list":[],"slot_precision":0, "slot_recall":0, "slot_f1":0}}

########## load slot labels ###############
labels = open("/data1/jiayu_xiao/project/wzh/MBart/text2text/slot_names.txt").read().split("\n")
label2idx = {}
for i in range(len(labels)):
    label2idx[labels[i].lower()] = i
label2idx["unk"] = i+1
label2idx["o"] = i+2
#print(label2idx)

labels_slot_tot = []
pred_slot_tot = []

for i in range(len(reference)):
    ref = reference[i].split("<s>")[0].strip().lower()
    gen = generated[i].split("<s>")[1].strip().lower()
    ref_full = reference_full[i].split("<intent-")[0].strip().lower()
    lang_ref = ref.split("lang-")[1].split(">")[0]
    lang_gen = gen.split("lang-")[1].split(">")[0] if "lang-" in gen else "$$"
    #print(ref,"=====",gen,"===",ref_full)
    #print(lang_ref,lang_gen)
    intent_ref = ref.split("intent-")[1].split(">")[0]
    intent_gen = gen.split("intent-")[1].split(">")[0] if "intent-" in gen else ""
    if lang_ref == lang_gen:
        dic[lang_ref]["lang_correct"] += 1
    dic[lang_ref]["lang_tot"] += 1
    if intent_gen == intent_ref:
        dic[lang_ref]["intent_correct"] += 1
    dic[lang_ref]["intent_tot"] += 1

    slot_ground_truth = [] ## the slot of full sentence
    sentence_ground_truth = [] ## the full sentence
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
        sentence_ground_truth.append(word)
        slot_ground_truth.append(slot_label)
        cnt += 1
    gen = gen.split("<intent-")[0].strip()
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
    #print(slot_ground_truth,"===",gen_slot_labels)
    assert len(slot_ground_truth) == len(gen_slot_labels)
    dic[lang_ref]["pred_list"].append(gen_slot_labels)
    dic[lang_ref]["ground_truth_list"].append(slot_ground_truth)


for lang in ["en","de","es","fr","zh"]:
    dic[lang]["lang_acc"] = dic[lang]["lang_correct"] / dic[lang]["lang_tot"]
    dic[lang]["intent_acc"] = dic[lang]["intent_correct"] / dic[lang]["intent_tot"]
    dic[lang]["slot_precision"] = precision_score(dic[lang]["pred_list"], dic[lang]["ground_truth_list"])
    dic[lang]["slot_recall"] = recall_score(dic[lang]["pred_list"], dic[lang]["ground_truth_list"])
    dic[lang]["slot_f1"] = f1_score(dic[lang]["pred_list"], dic[lang]["ground_truth_list"])
    print(lang,":","lang_acc:",dic[lang]["lang_acc"],"intent_acc:",dic[lang]["intent_acc"],"slot_f1:",dic[lang]["slot_f1"],)
    lang_acc_tot = 0.0
    intent_acc_tot = 0.0
    slot_f1 = 0.0

    for lang in dic.keys():
        lang_acc_tot += dic[lang]["lang_acc"]
        intent_acc_tot += dic[lang]["intent_acc"]
        slot_f1 += dic[lang]["slot_f1"]
    print("lang_acc_avg:",lang_acc_tot / len(dic.keys()),"intent_Acc_avg:",intent_acc_tot / len(dic.keys()), "slot_f1_avg:",slot_f1 / len(dic.keys()))

    
    
