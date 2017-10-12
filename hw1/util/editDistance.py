from enum import Enum
import Levenshtein


# Add transposition opt
def oneEditOpt(correct, raw):
    editOps = Levenshtein.editops(correct, raw)
    opsLength = len(editOps)
    if opsLength > 2 or opsLength == 0:
        return None
    elif opsLength == 1:
        return editOps[0]
    else:
        opt1 = editOps[0]
        opt2 = editOps[1]
        idx1 = opt1[1]
        idx2 = opt2[1]
        if opt1[0] == opt2[0] == 'replace' and abs(idx1 - idx2) == 1 and correct[idx1] == raw[idx2] and correct[idx2] == raw[idx1]:
            return 'transposition', idx1, idx2
        else:
            return None

def editOpts(correct, raw, threshold=2):
    opts = Levenshtein.editops(correct, raw)
    opsLength = len(opts)
    if opsLength > threshold or opsLength == 0:
        return None
    elif opsLength == 2:
        opt1 = opts[0]
        opt2 = opts[1]
        idx1 = opt1[1]
        idx2 = opt2[1]
        if opt1[0] == opt2[0] == 'replace' and abs(idx1 - idx2) == 1 and correct[idx1] == raw[idx2] and correct[idx2] == \
                raw[idx1]:
            return [('transposition', idx1, idx2)]
        else:
            return opts
    else:
        return opts



def normalizeOpt(correct, raw, opt):

    if not opt:
        return None

    optType = opt[0]
    idx1 = opt[1]
    idx2 = opt[2]


    if optType == 'delete':
        correctChar = correct[idx1]
        if idx1 == 0:
            previousChar = '>'
        else:
            previousChar = correct[idx1 - 1]
        return optType, previousChar + correctChar
    elif optType == 'replace':
        correctChar = correct[idx1]
        errorChar = raw[idx2]
        return optType, errorChar + correctChar
    elif optType == 'insert':
        previousChar = correct[idx1 - 1]
        errorChar = raw[idx2]
        return optType, previousChar + errorChar
    elif optType == 'transposition':
        previousChar = correct[idx1]
        postChar = raw[idx1]
        return optType, previousChar + postChar


if __name__ == '__main__':
    testSet = [('abc', 'abc'), ('abcd', 'acbd'), ('abc', 'abcd'), ('abc', 'acbd'), ('abc', 'adbc')]

    for pair in testSet:
        opt = oneEditOpt(*pair)
        print(pair, opt, normalizeOpt(*pair, opt))
