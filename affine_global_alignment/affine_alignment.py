import math

MATCH = 2
MISMATCH = -1
H = -2
G = -1


def s(a, b):
    return MATCH if a == b else MISMATCH


def align_affine(seqA, seqB):
    m, n = (len(seqA), len(seqB))
    M = [[-math.inf] * (n + 1) for _ in range(m + 1)]
    Ix = [[-math.inf] * (n + 1) for _ in range(m + 1)]
    Iy = [[-math.inf] * (n + 1) for _ in range(m + 1)]
    bpM = [[None] * (n + 1) for _ in range(m + 1)]
    bpX = [[None] * (n + 1) for _ in range(m + 1)]
    bpY = [[None] * (n + 1) for _ in range(m + 1)]
    M[0][0] = 0
    for i in range(1, m + 1):
        Ix[i][0] = H + i * G
        bpX[i][0] = ("M", i - 1, 0) if i == 1 else ("X", i - 1, 0)
    for j in range(1, n + 1):
        Iy[0][j] = H + j * G
        bpY[0][j] = ("M", 0, j - 1) if j == 1 else ("Y", 0, j - 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            candM = M[i - 1][j] + H + G
            candX = Ix[i - 1][j] + G
            candY = Iy[i - 1][j] + H + G
            vals = [candM, candX, candY]
            Ix[i][j] = max(vals)
            bpX[i][j] = [("M", i - 1, j), ("X", i - 1, j), ("Y", i - 1, j)][vals.index(Ix[i][j])]
            candM = M[i][j - 1] + H + G
            candX = Ix[i][j - 1] + H + G
            candY = Iy[i][j - 1] + G
            vals = [candM, candX, candY]
            Iy[i][j] = max(vals)
            bpY[i][j] = [("M", i, j - 1), ("X", i, j - 1), ("Y", i, j - 1)][vals.index(Iy[i][j])]
            best_prev, st = (M[i - 1][j - 1], "M")
            if Ix[i - 1][j - 1] > best_prev:
                best_prev, st = (Ix[i - 1][j - 1], "X")
            if Iy[i - 1][j - 1] > best_prev:
                best_prev, st = (Iy[i - 1][j - 1], "Y")
            M[i][j] = best_prev + s(seqA[i - 1], seqB[j - 1])
            bpM[i][j] = (st, i - 1, j - 1)
    end_scores = {"M": M[m][n], "X": Ix[m][n], "Y": Iy[m][n]}
    end_state = max(end_scores, key=end_scores.get)
    best_score = end_scores[end_state]
    i, j, state = (m, n, end_state)
    alnA, alnB, path = ([], [], [])
    while True:
        path.append((state, i, j))
        if state == "M":
            if i == 0 and j == 0:
                break
            prev = bpM[i][j]
            alnA.append(seqA[i - 1])
            alnB.append(seqB[j - 1])
        elif state == "X":
            prev = bpX[i][j]
            alnA.append(seqA[i - 1])
            alnB.append("-")
        else:
            prev = bpY[i][j]
            alnA.append("-")
            alnB.append(seqB[j - 1])
        state, i, j = prev
    alnA.reverse()
    alnB.reverse()
    path.reverse()
    stars = {(st, i, j) for (st, i, j) in path if not (i == 0 and j == 0)}
    bt = [p for p in reversed(path) if not (p[1] == 0 and p[2] == 0)]
    return {"score": best_score, "aligned_a": "".join(alnA), "aligned_b": "".join(alnB)}


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser(description="Affine-gap global alignment")
    p.add_argument("a")
    p.add_argument("b")
    x = p.parse_args()
    if any(c not in "ACGT" for c in x.a.upper() + x.b.upper()):
        p.error("Use A/C/G/T only")
    if (len(x.a) + 1) * (len(x.b) + 1) > 250000:
        p.error("Demo limited to 250000 DP cells")
    print(json.dumps(align_affine(x.a.upper(), x.b.upper()), indent=2))
