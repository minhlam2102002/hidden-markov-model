from algo import *
import numpy  as np

# Cac tham so de bai cho
size = 100
dices = np.array((0, 1))
numbers = np.array((0, 1, 2, 3, 4, 5))
pi = np.array((0.5, 0.5)) # initial distribution
transition = np.array(((0.8, 0.2), (0.3, 0.7)))
emissions = np.array(((1/6, 1/6, 1/6, 1/6, 1/6, 1/6), (0.1, 0.1, 0.1, 0.1, 0.1, 0.5)))
list_dices = np.zeros(([size, ]), dtype=int)
list_choices = np.zeros(([size, ]), dtype=int)

# b) sinh chuoi ngau nhien 100 mau
def generate():
    tmp_dice = np.random.choice(dices, p = pi)
    list_dices[0] = tmp_dice
    tmp_choice = np.random.choice(numbers, p = emissions[tmp_dice])
    list_choices[0] = tmp_choice
    for i in range(1, size):
        tmp_dice = np.random.choice(dices, p = transition[tmp_dice])
        list_dices[i] = tmp_dice
        tmp_choice = np.random.choice(numbers, p = emissions[tmp_dice])
        list_choices[i] = tmp_choice

# c) Dung Viterbi de du doan
def calc_accuracy():
    generate()
    predict = Viterbi(list_choices, transition, emissions, pi)
    count = 0
    for i in range(size):
        if predict[i] == list_dices[i]:
            count = count + 1
    return count/size
temp = []
N = 10
for k in range(N):
    temp.append(calc_accuracy())
    print(temp[k])
print("Average: ", sum(temp)/N)

# d) Dung Baum-Welch de tinh toan tham so
generate()
res_d = Baum_welch(list_choices, transition, emissions, pi)
print('\n',res_d['A'])
print(res_d['B'])