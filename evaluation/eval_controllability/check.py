import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--result_file",type=str,default='',)
args = parser.parse_args()

result_file = args.result_file
with open(result_file, "r") as f:
    result_data = json.load(f)

data = result_data['data']
meta = result_data['Meta_Info']

PN_count = 0
NP_count = 0
PP_list = []
PN_list = []
NP_list = []
NN_list = []
for data_i in data:
    scores_i = data_i['scores']

    if data_i['side'] == 'affirmative':
        PP_list.append(scores_i[0])
        PN_list.append(scores_i[1])
        if scores_i[0] == 100:
            continue
        else:
            PN_count += 1

    else:
        NP_list.append(scores_i[0])
        NN_list.append(scores_i[1])
        if scores_i[1] == 100:
            continue
        else:
            NP_count += 1

PP_value = sum(PP_list)/len(PP_list)
PN_value = sum(PN_list)/len(PN_list)
NP_value = sum(NP_list)/len(NP_list)
NN_value = sum(NN_list)/len(NN_list)

PP_v = [240-PN_count, PP_value]
PN_v = [PN_count, PN_value]
NP_v = [NP_count, NP_value]
NN_v = [240-NP_count, NN_value]

meta['PP_value'] = PP_v
meta['PN_value'] = PN_v
meta['NP_value'] = NP_v
meta['NN_value'] = NN_v

meta['CC_P_score'] = (240-PN_count)/240
meta['CC_N_score'] = (240-NP_count)/240
meta['CC_All_score'] = (480-PN_count-NP_count)/480

print(meta)

result_data['Meta_Info'] = meta

with open(result_file, "w") as f:
    json.dump(result_data, f, indent=4)
    pass
pass