import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rad_coverage import coverage_intervals, find_motifs, overlap_variants, union_length

ROOT = Path(__file__).resolve().parent


class RadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def cli(self, mode, enzyme=(), genome=None, vcf=None, extra=(), success=True):
        out = self.work / (mode + str(len(list(self.work.iterdir()))))
        result = subprocess.run([sys.executable, str(ROOT/'rad_coverage.py'),
                                 str(genome or ROOT/'examples/genome.fasta'),
                                 str(vcf or ROOT/'examples/variants.vcf'), mode, *enzyme,
                                 '--outdir', str(out), *extra], capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        if not success:
            self.assertFalse(out.exists())
            return None, out
        return json.loads((out/'summary.json').read_text()), out

    def test_all_three_modes_known_outputs(self):
        for mode, enzymes, expected in [
            ('SingleRad', ('EcoRI',), (3,600,3,3)),
            ('ddRad', ('EcoRI','--enzyme2','HindIII'), (8,800,4,4)),
            ('2bRad', (), (1,34,1,1))]:
            with self.subTest(mode=mode):
                summary,out=self.cli(mode,enzymes)
                self.assertEqual(tuple(summary[k] for k in ['interval_count','covered_bases_union','covered_variant_records','intervals_with_variants']),expected)
                self.assertEqual(summary['eligible_variant_records'],6)
                intervals=[line.split('\t') for line in (out/'intervals.bed').read_text().splitlines()]
                self.assertEqual(len(intervals),expected[0])
                self.assertEqual(len((out/'covered_variants.tsv').read_text().splitlines())-1,expected[2])
                if mode=='2bRad':self.assertEqual(intervals,[['synthetic_chr1','600','634']])

    def test_overlapping_motifs_and_iupac(self):
        self.assertEqual(find_motifs('AAAAA','AAA'),[0,1,2])
        self.assertEqual(find_motifs('AAGAC','AR'),[0,1])
        self.assertEqual(find_motifs('ANN','ANN'),[])

    def test_bed_vcf_boundary_conversion_and_union(self):
        variants=[{'pos':p} for p in [100,101,200,201]]
        counts,covered=overlap_variants([(100,200)],variants)
        self.assertEqual(counts,[2])
        self.assertEqual([x['pos'] for x in covered],[101,200])
        self.assertEqual(union_length([(0,200),(100,300),(300,310)]),310)

    def test_variant_counts_all_overlapping_intervals_once_per_record(self):
        counts,covered=overlap_variants([(0,200),(100,300)],[{'pos':150}])
        self.assertEqual(counts,[1,1])
        self.assertEqual(len(covered),1)

    def test_ddrad_size_boundaries_and_intervening_first_enzyme(self):
        for distance,expected in [(299,0),(300,2),(700,2),(701,0)]:
            dna=list('A'*1000)
            dna[100:106]='GAATTC';dna[100+distance:106+distance]='AAGCTT'
            self.assertEqual(len(coverage_intervals(''.join(dna),'ddRad','EcoRI','HindIII')),expected)
        dna=list('A'*800)
        for pos,motif in [(100,'GAATTC'),(200,'GAATTC'),(450,'AAGCTT')]:dna[pos:pos+6]=motif
        self.assertEqual(coverage_intervals(''.join(dna),'ddRad','EcoRI','HindIII'),[])

    def test_gzip_phased_genotypes_and_chromosome_selection(self):
        vcf=self.work/'variants.vcf.gz'
        with gzip.open(vcf,'wt') as stream:
            stream.write((ROOT/'examples/variants.vcf').read_text().replace('0/0','0|0').replace('1/1','1|1'))
        summary,_=self.cli('SingleRad',('EcoRI',),vcf=vcf)
        self.assertEqual(summary['covered_variant_records'],3)
        summary,_=self.cli('SingleRad',('EcoRI',),vcf=vcf,extra=('--chrom','other_chr'))
        self.assertEqual(summary['sequence_length'],200)
        self.assertEqual(summary['eligible_variant_records'],1)
        self.assertEqual(summary['interval_count'],0)

    def test_invalid_inputs_fail_before_output(self):
        self.cli('ddRad',('EcoRI',),success=False)
        self.cli('SingleRad',('NotAnEnzyme',),success=False)
        self.cli('2bRad',extra=('--chrom','missing'),success=False)
        vcf=self.work/'bad.vcf';vcf.write_text('not a VCF\n')
        self.cli('SingleRad',('EcoRI',),vcf=vcf,success=False)

    def test_repeatability(self):
        a,da=self.cli('ddRad',('EcoRI','--enzyme2','HindIII'))
        b,db=self.cli('ddRad',('EcoRI','--enzyme2','HindIII'))
        self.assertEqual(a,b)
        for name in ['intervals.bed','variant_intervals.bed','covered_variants.tsv','summary.json']:
            self.assertEqual((da/name).read_bytes(),(db/name).read_bytes())


if __name__=='__main__':unittest.main()
