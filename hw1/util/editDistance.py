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

    transOpts = []
    if opsLength == 0:
        transOpts = None
    elif opsLength >= 2:
        idx = 0
        while idx < opsLength - 1:
            opt1 = opts[idx]
            opt2 = opts[idx + 1]
            idx1 = opt1[2]
            idx2 = opt2[2]
            try:
                if opt1[0] == opt2[0] == 'replace' and abs(idx1 - idx2) == 1 and correct[idx1] == raw[idx2] and correct[idx2] == raw[idx1]:
                    transOpts.append(('transposition', idx1, idx2))
                    idx += 2
                else:
                    transOpts.append(opt1)
                    idx += 1
            except:
                transOpts.append(opt1)
                idx += 1

        if idx == opsLength - 1:
            transOpts.append(opts[idx])
    else:
        transOpts = opts

    if transOpts and len(transOpts) > threshold:
        return None
    else:
        return transOpts


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
    testSet = [('PADDING_TOKEN', 'protectionst'), ('abc', 'abc'), ('abcd', 'acbd'), ('abc', 'abcde'), ('abcde', 'acbdd'), ('abc', 'adbc')]

    for pair in testSet:
        opts = editOpts(*pair)
        print(opts)
        if opts is not None:
            print(pair, opts, list(map(lambda opt: normalizeOpt(*pair, opt), opts)))
        else:
            print(pair, opts, None)
        print('\n')
