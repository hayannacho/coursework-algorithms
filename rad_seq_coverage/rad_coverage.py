#!/usr/bin/env python3
"""Educational RAD coverage windows and two-sample VCF overlap, not a digestion simulator."""
import argparse
from bisect import bisect_left, bisect_right
import csv
import gzip
import json
from pathlib import Path
import re

from Bio import SeqIO
from Bio.Data.IUPACData import ambiguous_dna_values
from Bio.Restriction.Restriction_Dictionary import rest_dict

TWO_B_MOTIF = "NNNNNNNNNNCGANNNNNNTGCNNNNNNNNNNNN"


def find_motifs(sequence, motif):
    """Forward-reference, overlapping IUPAC motif starts, in 0-based coordinates."""
    try:
        pattern = "".join("[" + ambiguous_dna_values[base] + "]" for base in motif.upper())
    except KeyError as error:
        raise ValueError(f"Unsupported motif symbol: {error.args[0]}") from error
    if not motif:
        raise ValueError("Motif must not be empty")
    return [match.start() for match in re.finditer(f"(?={pattern})", sequence.upper())]


def enzyme_motif(name):
    if name not in rest_dict:
        raise ValueError(f"Unknown restriction enzyme: {name}")
    return rest_dict[name]["site"]


def coverage_intervals(sequence, mode, enzyme1=None, enzyme2=None):
    """Return unique 0-based half-open windows under the documented coursework model."""
    size = len(sequence)
    if mode == "2bRad":
        return [(start, start + len(TWO_B_MOTIF)) for start in find_motifs(sequence, TWO_B_MOTIF)]
    first = find_motifs(sequence, enzyme_motif(enzyme1))
    if mode == "SingleRad":
        return sorted({(max(0, start - 100), min(size, start + 100)) for start in first})
    if mode != "ddRad":
        raise ValueError(f"Unknown mode: {mode}")
    if enzyme1 == enzyme2:
        raise ValueError("ddRad requires two different enzyme names")
    second = find_motifs(sequence, enzyme_motif(enzyme2))
    windows = set()
    # Preserve the original course model: RE2 sites at 300..700 bases, with no intervening RE1.
    # This is deliberately not a full adjacent-cut, both-strand digestion model.
    for index, anchor in enumerate(first):
        previous = first[index - 1] if index else -1
        following = first[index + 1] if index + 1 < len(first) else size
        for other in second[bisect_left(second, anchor + 300):bisect_right(second, anchor + 700)]:
            if other < following:
                windows.update(((anchor, min(size, anchor + 100)), (max(0, other - 100), other)))
        for other in second[bisect_left(second, anchor - 700):bisect_right(second, anchor - 300)]:
            if other > previous:
                windows.update(((other, min(size, other + 100)), (max(0, anchor - 100), anchor)))
    return sorted(windows)


def read_sequence(path, chromosome=None):
    for record in SeqIO.parse(path, "fasta"):
        if chromosome is None or record.id == chromosome:
            sequence = str(record.seq).upper()
            if not sequence or set(sequence) - set("ACGTN"):
                raise ValueError("Selected FASTA record must contain nonempty A/C/G/T/N sequence")
            return record.id, sequence
    raise ValueError("No matching FASTA record found")


def eligible_variants(path, chromosome, size):
    """Keep selected-chromosome VCF rows with two opposite homozygous diploid calls."""
    opener = gzip.open if path.suffix == ".gz" else open
    variants = []
    header_seen = False
    with opener(path, "rt") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.startswith("##") or not line.strip():
                continue
            columns = line.rstrip("\n\r").split("\t")
            if line.startswith("#CHROM\t"):
                if len(columns) != 11:
                    raise ValueError("VCF must have exactly two sample columns")
                header_seen = True
                continue
            if line.startswith("#"):
                continue
            if not header_seen or len(columns) != 11:
                raise ValueError(f"Malformed VCF record/header at line {line_number}")
            if columns[0] != chromosome:
                continue
            position = int(columns[1])
            if not 1 <= position <= size:
                raise ValueError(f"VCF position outside selected FASTA record at line {line_number}")
            formats = columns[8].split(":")
            if "GT" not in formats:
                continue
            index = formats.index("GT")
            calls = [sample.split(":") for sample in columns[9:11]]
            if any(index >= len(call) for call in calls):
                continue
            genotypes = {call[index].replace("|", "/") for call in calls}
            if genotypes == {"0/0", "1/1"}:
                variants.append({"chrom": chromosome, "pos": position, "id": columns[2],
                                 "ref": columns[3], "alt": columns[4], "source_line": line_number})
    if not header_seen:
        raise ValueError("VCF #CHROM header is missing")
    return sorted(variants, key=lambda row: (row["pos"], row["source_line"]))


def overlap_variants(intervals, variants):
    positions = [row["pos"] for row in variants]
    interval_counts = []
    covered_indices = set()
    for start, end in intervals:
        # VCF POS is 1-based: start < POS <= end for a BED interval [start, end).
        left, right = bisect_right(positions, start), bisect_right(positions, end)
        interval_counts.append(right - left)
        covered_indices.update(range(left, right))
    return interval_counts, [variants[index] for index in sorted(covered_indices)]


def union_length(intervals):
    total, stop = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, stop))
        stop = max(stop, end)
    return total


def write_outputs(outdir, chromosome, intervals, counts, variants, summary):
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "intervals.bed").open("w") as all_regions, (outdir / "variant_intervals.bed").open("w") as matched:
        for (start, end), count in zip(intervals, counts):
            row = f"{chromosome}\t{start}\t{end}\n"
            all_regions.write(row)
            if count:
                matched.write(row)
    with (outdir / "covered_variants.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["chrom", "pos", "id", "ref", "alt"], delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(variants)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("genome", type=Path)
    parser.add_argument("vcf", type=Path)
    parser.add_argument("mode", choices=["SingleRad", "ddRad", "2bRad"])
    parser.add_argument("enzyme1", nargs="?")
    parser.add_argument("--enzyme2")
    parser.add_argument("--chrom", help="FASTA ID / VCF CHROM; default is the first FASTA record")
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    if args.mode != "2bRad" and not args.enzyme1:
        parser.error("SingleRad/ddRad requires enzyme1")
    if args.mode == "ddRad" and not args.enzyme2:
        parser.error("ddRad requires --enzyme2")
    if args.mode == "2bRad" and (args.enzyme1 or args.enzyme2):
        parser.error("2bRad uses the fixed coursework motif; omit enzyme names")
    if args.mode == "SingleRad" and args.enzyme2:
        parser.error("--enzyme2 is only valid for ddRad")
    try:
        chromosome, sequence = read_sequence(args.genome, args.chrom)
        intervals = coverage_intervals(sequence, args.mode, args.enzyme1, args.enzyme2)
        eligible = eligible_variants(args.vcf, chromosome, len(sequence))
        counts, covered = overlap_variants(intervals, eligible)
        summary = {"model": "coursework_forward_recognition_start", "mode": args.mode,
                   "chromosome": chromosome, "sequence_length": len(sequence),
                   "interval_count": len(intervals), "covered_bases_union": union_length(intervals),
                   "eligible_variant_records": len(eligible), "covered_variant_records": len(covered),
                   "intervals_with_variants": sum(count > 0 for count in counts),
                   "bed_coordinates": "0-based, half-open", "variant_positions": "1-based VCF POS"}
        output_paths = {(args.outdir / name).resolve() for name in ["intervals.bed", "variant_intervals.bed", "covered_variants.tsv", "summary.json"]}
        if args.genome.resolve() in output_paths or args.vcf.resolve() in output_paths:
            raise ValueError("Output must not overwrite an input file")
        write_outputs(args.outdir, chromosome, intervals, counts, covered, summary)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
