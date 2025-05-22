import numpy as np



def PREPROCESS_cas9(lines):
    data_n = len(lines) - 1
    SEQ = np.zeros((data_n, 23, 4), dtype=int)
    label = np.zeros((data_n, 1, 1))

    for l in range(1, data_n + 1):
        data = lines[l].split(',')

        y = float(data[2])
        if y < 0:
            label[l - 1, 0, 0] = 0
        else:
            label[l - 1, 0, 0] = y

        seq = data[1]
        for i in range(23):
            if seq[i] in "Aa":
                SEQ[l - 1, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l - 1, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l - 1, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l - 1, i, 3] = 1

    return SEQ, label


def PREPROCESS_for_DeepCRISPR(lines):
    data_n = len(lines) - 1
    SEQ = np.zeros((data_n, 1, 23, 4), dtype=int)
    label = np.zeros((data_n, 1, 1, 1))

    for l in range(1, data_n + 1):
        data = lines[l].split(',')

        y = float(data[2])
        if y < 0:
            label[l - 1, 0, 0, 0] = 0
        else:
            label[l - 1, 0, 0, 0] = y

        seq = data[1]
        for i in range(23):
            if seq[i] == "A":
                SEQ[l - 1, 0, i, 0] = 1
            elif seq[i] == "C":
                SEQ[l - 1, 0, i, 1] = 1
            elif seq[i] == "G":
                SEQ[l - 1, 0, i, 2] = 1
            elif seq[i] == "T":
                SEQ[l - 1, 0, i, 3] = 1

    return SEQ, label



def PREPROCESS_cas12(lines):
    data_n = len(lines) - 1
    SEQ = np.zeros((data_n, 34, 4), dtype=int)
    label = np.zeros((data_n, 1, 1))

    for l in range(1, data_n + 1):
        data = lines[l].split(',')

        y = float(data[2])
        if y < 0:
            label[l - 1, 0, 0] = 0
        else:
            label[l - 1, 0, 0] = y

        seq = data[1]
        for i in range(34):
            if seq[i] in "Aa":
                SEQ[l - 1, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l - 1, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l - 1, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l - 1, i, 3] = 1

    return SEQ, label




