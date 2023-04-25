import json 
##0.8555431131019037
##dic:0.7883538633818589
reference = open("./MultiATIS_data/processed_intent/test_EN.output").read().split("\n")[:-1]
generated = json.load(open("./outputs/prompt_intent/_epoch0/en.txt","r"))
intent_correct = 0
intent_tot = 0
for i in range(len(reference)):
    ref = reference[i].strip().lower()
    gen = generated[i].strip().lower()
    intent_ref = ref.split("@")[1].strip()
    intent_gen = gen.split("@")[1].strip()
    print(intent_gen,intent_ref)
    if intent_ref == intent_gen:
        intent_correct += 1
    intent_tot += 1
        
print(intent_correct / intent_tot)
