#!/usr/bin/env python3
"""Count optimal local DNA-alignment paths using score and count dynamic-programming tables."""

def alignment_summary(x, y, match=2):
    """Count coordinate-distinct optimal paths with strictly positive prefix scores."""
    if match not in (1, 2):
        raise ValueError("match must be 1 (archive) or 2 (assignment)")
    if any(base not in "ACGT" for base in x + y):
        raise ValueError("Sequences must contain only uppercase A/C/G/T; empty input is allowed")
    rows = len(x) + 1
    cols = len(y) + 1
    
    MATCH = match
    MISMATCH = -1
    GAP = -1
    
    # H: score matrix
    H = [[0] * cols for _ in range(rows)]
    # C: count matrix
    C = [[0] * cols for _ in range(rows)]

    # Fill DP
    for i in range(1, rows):
        xi = x[i - 1]
        for j in range(1, cols):
            yj = y[j - 1]
            s = MATCH if xi == yj else MISMATCH

            diag = H[i - 1][j - 1] + s
            up   = H[i - 1][j] + GAP
            left = H[i][j - 1] + GAP

            hij = max(0, diag, up, left)
            H[i][j] = hij

            if hij == 0:
                C[i][j] = 0
                continue

            total = 0
            # From diagonal
            if diag == hij:
                if H[i - 1][j - 1] == 0 and s == MATCH:
                    total += 1  # start new alignment
                if H[i - 1][j - 1] > 0:
                    total += C[i - 1][j - 1]

            # From up
            if up == hij and H[i - 1][j] > 0:
                total += C[i - 1][j]

            # From left
            if left == hij and H[i][j - 1] > 0:
                total += C[i][j - 1]

            C[i][j] = total

    # Find maximum score
    max_score = 0
    for i in range(rows):
        for j in range(cols):
            if H[i][j] > max_score:
                max_score = H[i][j]

    if max_score == 0:
        return {"score": 0, "count": 0, "endpoints": []}

    # Sum counts for all cells with max score
    total_optimal = 0
    for i in range(rows):
        for j in range(cols):
            if H[i][j] == max_score:
                total_optimal += C[i][j]

    endpoints = [{"x_end": i, "y_end": j, "count": C[i][j]}
                 for i in range(rows) for j in range(cols) if H[i][j] == max_score]
    return {"score": max_score, "count": total_optimal, "endpoints": endpoints}


def count_optimal_alignments(x, y, match=2):
    return alignment_summary(x, y, match)["count"]


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", help="First DNA sequence (ACGT; empty string allowed)")
    parser.add_argument("y", help="Second DNA sequence")
    parser.add_argument("--match", type=int, choices=[1, 2], default=2,
                        help="2: assignment scoring; 1: archived implementation scoring")
    args = parser.parse_args()
    # Full score/count matrices are pedagogical, not genome-scale storage.
    if (len(args.x) + 1) * (len(args.y) + 1) > 1_000_000:
        parser.error("Demo limited to 1,000,000 DP cells")
    try:
        result = alignment_summary(args.x.upper(), args.y.upper(), args.match)
    except ValueError as error:
        parser.error(str(error))
    result.update({"match": args.match, "mismatch": -1, "gap": -1,
                   "count_definition": "coordinate-distinct positive-score traceback paths",
                   "endpoint_coordinates": "0-based exclusive ends"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
