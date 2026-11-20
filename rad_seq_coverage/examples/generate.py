"""Generate artificial FASTA/VCF fixtures; no biological sample data."""
from pathlib import Path

root = Path(__file__).resolve().parent
dna = list("A" * 1800)
for position, motif in [(100, "GAATTC"), (800, "GAATTC"), (1500, "GAATTC"),
                        (450, "AAGCTT"), (1150, "AAGCTT"), (610, "CGAAAAAAATGC")]:
    dna[position:position+len(motif)] = motif
(root / "genome.fasta").write_text(">synthetic_chr1\n" + "".join(dna) + "\n>other_chr\n" + "A"*200 + "\n")
rows = [(150,"0/0","1/1"),(400,"1/1","0/0"),(620,"0/0","1/1"),
        (850,"0/0","1/1"),(1450,"0/0","1/1"),(1550,"0/1","0/0"),
        (1100,"./.","1/1"),(1700,"0/0","1/1")]
header='##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsynthetic_A\tsynthetic_B\n'
with (root / "variants.vcf").open("w") as out:
    out.write(header)
    for index,(pos,a,b) in enumerate(rows,1):
        out.write(f"synthetic_chr1\t{pos}\tv{index}\t{dna[pos-1]}\tC\t.\tPASS\t.\tGT\t{a}\t{b}\n")
    out.write("other_chr\t100\tother\tA\tC\t.\tPASS\t.\tGT\t0/0\t1/1\n")
