def valid_DNA_sequence(DNA):
    s = "".join((ch for ch in DNA if not ch.isspace()))
    if s == "":
        return False
    for ch in s:
        if ch not in "AaCcGgTt":
            return False
    return True


def translate(DNA_sequence, mode):
    codontable = {
        "ATA": "I",
        "ATC": "I",
        "ATT": "I",
        "ATG": "M",
        "ACA": "T",
        "ACC": "T",
        "ACG": "T",
        "ACT": "T",
        "AAC": "N",
        "AAT": "N",
        "AAA": "K",
        "AAG": "K",
        "AGC": "S",
        "AGT": "S",
        "AGA": "R",
        "AGG": "R",
        "CTA": "L",
        "CTC": "L",
        "CTG": "L",
        "CTT": "L",
        "CCA": "P",
        "CCC": "P",
        "CCG": "P",
        "CCT": "P",
        "CAC": "H",
        "CAT": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGA": "R",
        "CGC": "R",
        "CGG": "R",
        "CGT": "R",
        "GTA": "V",
        "GTC": "V",
        "GTG": "V",
        "GTT": "V",
        "GCA": "A",
        "GCC": "A",
        "GCG": "A",
        "GCT": "A",
        "GAC": "D",
        "GAT": "D",
        "GAA": "E",
        "GAG": "E",
        "GGA": "G",
        "GGC": "G",
        "GGG": "G",
        "GGT": "G",
        "TCA": "S",
        "TCC": "S",
        "TCG": "S",
        "TCT": "S",
        "TTC": "F",
        "TTT": "F",
        "TTA": "L",
        "TTG": "L",
        "TAC": "Y",
        "TAT": "Y",
        "TAA": "-",
        "TAG": "-",
        "TGC": "C",
        "TGT": "C",
        "TGA": "-",
        "TGG": "W",
    }
    stop_codons = {"TAA", "TAG", "TGA"}
    if mode == "VERBOSE":
        tokens = []
        for i in range(0, len(DNA_sequence), 3):
            codon = DNA_sequence[i : i + 3].upper()
            aa1 = codontable.get(codon, "-")
            if codon in stop_codons:
                tokens.append("Stop")
            elif codon == "ATG":
                tokens.append("Met")
            else:
                tokens.append(aa1)
        return " ".join(tokens)
    out_seq = ""
    for i in range(0, len(DNA_sequence), 3):
        codon = DNA_sequence[i : i + 3].upper()
        aa = codontable.get(codon, "-")
        if mode == "DNA":
            if codon in stop_codons:
                token = " - "
            elif codon == "ATG":
                token = " M "
            else:
                token = f" {aa} "
        else:
            token = aa
        out_seq += token
    return out_seq


def reverse_complement(DNA_sequence):
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "a": "t", "t": "a", "c": "g", "g": "c"}
    s = "".join((ch for ch in DNA_sequence if not ch.isspace()))
    rc = "".join((comp.get(bp, bp) for bp in s[::-1]))
    return rc.upper()


def six_frames(sequence):
    if not valid_DNA_sequence(sequence):
        raise ValueError("Expected nonempty A/C/G/T sequence")
    sequence = "".join(sequence.split()).upper()
    result = {}
    for strand, seq in [("+", sequence), ("-", reverse_complement(sequence))]:
        for frame in range(3):
            fragment = seq[frame:]
            fragment = fragment[: len(fragment) // 3 * 3]
            result[strand + str(frame + 1)] = translate(fragment, "COMPACT").replace("-", "*")
    return result


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser(description="Six-frame translation using the standard genetic code")
    p.add_argument("sequence")
    a = p.parse_args()
    try:
        print(json.dumps(six_frames(a.sequence), indent=2))
    except ValueError as e:
        p.error(str(e))
