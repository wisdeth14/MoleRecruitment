import numpy as np
import statistics

# moles=False
f = open('logs\imagenet\ssre\imagenet100_molesFalse_seed4_rho0.5.txt', 'r')
contents_baseline = f.read()
f.close()
baseline_string = contents_baseline.split('*')[-1]

# moles=True
f = open('logs\imagenet\ssre\imagenet100_molesTrue_seed4_rho0.01.txt', 'r')
contents_attacked = f.read()
f.close()
attacked_string = contents_attacked.split('*')[-1]

filter_value = 10.0 #filter out classes that have sub 10% baseline acc

baselineAcc = {}
attackedAcc = {}
confoundingAcc = {}
moleAcc = {}
confounding = {}

baseline = baseline_string.split('|')[:-1]
for i in baseline:
    c = i.split(' - ')[0].strip()
    acc = float(i.split(' - ')[1].strip())
    baselineAcc[c] = acc

mole_acc = []
moles = attacked_string.split('ATTACKED CLASSES')[0].split('|')[:-1]
for i in moles:
    mole_acc.append(float(i.split(' - ')[1].strip()))
    moleAcc[i.split(' - ')[0].strip()] = float(i.split(' - ')[1].strip())

attacked_string = attacked_string.replace('"', '')
attacked = attacked_string.split('ATTACKED CLASSES')[1].split('|')[:-1]
for i in attacked:
    c = i.split(' - ')[0].strip()
    if baselineAcc[c] > filter_value:
        acc = float(i.split(' - ')[1].split('confounding')[0].strip())
        classes = i.split('[')[1].split(']')[0].split(',')
        for j in classes:
            confoundingAcc[j.strip().strip("'")] = moleAcc[j.strip().strip("'")]
        attackedAcc[c] = acc
        confounding[c] = len(classes)

attacked_ratio = []
for c in attackedAcc.keys():
    if baselineAcc[c] > filter_value:
        attacked_ratio.append((attackedAcc[c] - baselineAcc[c]) / baselineAcc[c])
        #print(c, (attackedAcc[c] - baselineAcc[c]) / baselineAcc[c])
confounding_ratio = []
for c in confoundingAcc.keys():
    if baselineAcc[c] > filter_value:
        confounding_ratio.append((confoundingAcc[c] - baselineAcc[c]) / baselineAcc[c])

print(np.sort(attacked_ratio))


total_accuracy_baseline = np.mean(list(baselineAcc.values()))
total_accuracy_moles = sum(moleAcc.values()) / len(moleAcc.values())
total_accuracy_attacked = np.mean(list(attackedAcc.values()))
total_accuracy_baseline_justattacked = np.mean([baselineAcc[key] for key in attackedAcc.keys()])
total_accuracy_confounding = np.mean(list(confoundingAcc.values()))
total_accuracy_baseline_justconfounding = np.mean([baselineAcc[key] for key in confoundingAcc.keys()])

# print(total_accuracy_baseline)
# print(total_accuracy_moles)
# print(total_accuracy_baseline_justattacked)
# print(total_accuracy_attacked)

print('total baseline: ', np.round(total_accuracy_baseline, 1))
print('# attacked: ', len(attackedAcc.values()))
print('# confounding: ', np.sum(list(confounding.values())))
print('delta attacked: ', np.round((total_accuracy_attacked - total_accuracy_baseline_justattacked), 1))
print('delta attacked / attacked (median): ', np.round(statistics.median(attacked_ratio) * 100, 1))
print('delta attacked / attacked (mean): ', np.round(statistics.mean(attacked_ratio) * 100, 1))
print('delta attacked / attacked (stdev): ', np.round(statistics.stdev(attacked_ratio) * 100, 1))
print('delta total: ', np.round((total_accuracy_moles - total_accuracy_baseline), 1))
print('delta total / total: ', np.round((total_accuracy_moles - total_accuracy_baseline) / total_accuracy_baseline * 100, 1))
# print('delta confounding: ', np.round((total_accuracy_confounding - total_accuracy_baseline_justconfounding), 1))
# print('delta confounding / confounding (median): ', np.round(statistics.median(confounding_ratio) * 100, 1))

# print('delta attacked: ', np.round((total_accuracy_attacked - total_accuracy_baseline_justattacked), 1))
# print('delta attacked / attacked: ', np.round((total_accuracy_attacked - total_accuracy_baseline_justattacked) / total_accuracy_baseline_justattacked * 100, 1))
# print('delta total: ', np.round((total_accuracy_moles - total_accuracy_baseline), 1))
# print('delta total / total: ', np.round((total_accuracy_moles - total_accuracy_baseline) / total_accuracy_baseline * 100, 1))