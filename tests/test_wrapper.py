"""
Tests for the Wrapper module
"""

from __future__ import annotations

import unittest

import pandas as pd
from scipy import stats

from gnatpy._datagen import _generate_rank_entropy_anndata
from gnatpy.wrapper import rank_entropy, rank_entropy_comparison


class TestRankEntropy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = _generate_rank_entropy_anndata(
            n_ordered_samples=15,
            n_unordered_samples=20,
            n_ordered_genes=13,
            n_unordered_genes=5,
            dist=stats.norm(20, 10),
            noise_dist=stats.norm(0, 1),
            noise_swaps=15,
            shuffle_genes=True,
            shuffle_samples=True,
            seed=19283019283091283,
        )

    def method_tests(self, method):
        rank_entropy_df = rank_entropy(
            data=self.test_data,
            sample_groups={
                "ordered": (self.test_data.obs["sample_type"] == "ordered"),
                "unordered": (self.test_data.obs["sample_type"] == "unordered"),
            },
            gene_networks={
                "ordered": (self.test_data.var["gene_type"] == "ordered"),
                "unordered": (self.test_data.var["gene_type"] == "unordered"),
            },
            method=method,
            layer=None,  # Should be just the default
            processes=1,  # Don't try to split across processes yet
        )
        # Should be a dataframe
        self.assertIsInstance(rank_entropy_df, pd.DataFrame)
        # Should have ordered and unordered in its index
        self.assertCountEqual(rank_entropy_df.index, ["ordered", "unordered"])
        self.assertCountEqual(rank_entropy_df.columns, ["ordered", "unordered"])
        # There shouldn't be any NaN in this dataframe
        self.assertFalse(rank_entropy_df.isna().any().any())  # type: ignore
        # The rank entropy of the ordered sample/gene set should be
        # less than the rank entropy of the disorded sample set
        self.assertLess(
            rank_entropy_df.loc["ordered", "ordered"],
            rank_entropy_df.loc["ordered", "unordered"],
        )
        self.assertLess(
            rank_entropy_df.loc["ordered", "ordered"],
            rank_entropy_df.loc["unordered", "ordered"],
        )

    def test_dirac(self):
        self.method_tests("DIRAC")

    def test_infer(self):
        self.method_tests("INFER")

    def test_race(self):
        self.method_tests("RACE")

    def test_crane(self):
        self.method_tests("CRANE")

    def parallel_tests(self, method):
        serial_rank_entropy_df = rank_entropy(
            data=self.test_data,
            sample_groups={
                "ordered": (self.test_data.obs["sample_type"] == "ordered"),
                "unordered": (self.test_data.obs["sample_type"] == "unordered"),
            },
            gene_networks={
                "ordered": (self.test_data.var["gene_type"] == "ordered"),
                "unordered": (self.test_data.var["gene_type"] == "unordered"),
            },
            method=method,
            layer=None,  # Should be just the default
            processes=1,  # Don't try to split across processes yet
        )
        parallel_rank_entropy_df = rank_entropy(
            data=self.test_data,
            sample_groups={
                "ordered": (self.test_data.obs["sample_type"] == "ordered"),
                "unordered": (self.test_data.obs["sample_type"] == "unordered"),
            },
            gene_networks={
                "ordered": (self.test_data.var["gene_type"] == "ordered"),
                "unordered": (self.test_data.var["gene_type"] == "unordered"),
            },
            method=method,
            layer=None,  # Should be just the default
            processes=2,  # Don't try to split across processes yet
        )
        pd.testing.assert_frame_equal(serial_rank_entropy_df, parallel_rank_entropy_df)

    def test_dirac_parallel(self):
        self.parallel_tests("DIRAC")

    def test_race_parallel(self):
        self.parallel_tests("RACE")

    def test_infer_parallel(self):
        self.parallel_tests("INFER")

    def test_crane_parallel(self):
        self.parallel_tests("CRANE")


class TestRankEntropyComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = _generate_rank_entropy_anndata(
            n_ordered_samples=15,
            n_unordered_samples=20,
            n_ordered_genes=13,
            n_unordered_genes=5,
            dist=stats.norm(20, 10),
            noise_dist=stats.norm(0, 1),
            noise_swaps=15,
            shuffle_genes=True,
            shuffle_samples=True,
            seed=19283019283091283,
        )

    def method_tests(self, method, **kwargs):
        rank_entropy_stat, rank_entropy_pvalue = rank_entropy_comparison(
            data=self.test_data,
            sample_groups=[
                (self.test_data.obs["sample_type"] == "ordered"),
                (self.test_data.obs["sample_type"] == "unordered"),
            ],
            gene_networks={
                "ordered": (self.test_data.var["gene_type"] == "ordered"),
                "unordered": (self.test_data.var["gene_type"] == "unordered"),
            },
            method=method,
            layer=None,  # Should be just the default
            seed=8293048029375,
            processes=1,  # Don't try to split across processes yet
            **kwargs,
        )
        # Check the expected datatypes
        self.assertIsInstance(rank_entropy_stat, pd.Series)
        self.assertIsInstance(rank_entropy_pvalue, pd.Series)
        # They should both be length 2
        self.assertEqual(len(rank_entropy_stat), 2)
        self.assertEqual(len(rank_entropy_pvalue), 2)
        # They should have 'ordered', and 'unordered' index values
        self.assertCountEqual(rank_entropy_stat.index, ["ordered", "unordered"])
        self.assertCountEqual(rank_entropy_pvalue.index, ["ordered", "unordered"])
        # The ordered gene network should have a stat greater than 0,
        # and a p-value of less than 0.05
        self.assertGreater(rank_entropy_stat["ordered"], 0)
        self.assertLess(rank_entropy_pvalue["ordered"], 0.05)
        # The unordered gene network should have a p-value greater than 0.05
        # (and hopefully much greater than it)
        self.assertGreater(rank_entropy_pvalue["unordered"], 0.2)

    def test_dirac(self):
        self.method_tests("DIRAC")

    def test_infer(self):
        self.method_tests("INFER")

    def test_race(self):
        self.method_tests("RACE", iterations=100)

    def test_crane(self):
        self.method_tests("CRANE")

    def method_tests_parallel(self, method, **kwargs):
        serial_rank_entropy_stat, serial_rank_entropy_pvalue = rank_entropy_comparison(
            data=self.test_data,
            sample_groups=[
                (self.test_data.obs["sample_type"] == "ordered"),
                (self.test_data.obs["sample_type"] == "unordered"),
            ],
            gene_networks={
                "ordered": (self.test_data.var["gene_type"] == "ordered"),
                "unordered": (self.test_data.var["gene_type"] == "unordered"),
            },
            method=method,
            layer=None,  # Should be just the default
            seed=8293048029375,
            processes=1,  # Don't try to split across processes yet
            **kwargs,
        )
        parallel_rank_entropy_stat, parallel_rank_entropy_pvalue = (
            rank_entropy_comparison(
                data=self.test_data,
                sample_groups=[
                    (self.test_data.obs["sample_type"] == "ordered"),
                    (self.test_data.obs["sample_type"] == "unordered"),
                ],
                gene_networks={
                    "ordered": (self.test_data.var["gene_type"] == "ordered"),
                    "unordered": (self.test_data.var["gene_type"] == "unordered"),
                },
                method=method,
                layer=None,  # Should be just the default
                seed=8293048029375,
                processes=2,  # Don't try to split across processes yet
                **kwargs,
            )
        )
        # The stat values should be the same
        pd.testing.assert_series_equal(
            serial_rank_entropy_stat, parallel_rank_entropy_stat
        )
        # The p-values should be about the same (the seeding
        # works slightly differently between the two modes unfortunately)
        pd.testing.assert_series_equal(
            serial_rank_entropy_pvalue, parallel_rank_entropy_pvalue
        )

    def test_dirac_parallel(self):
        self.method_tests_parallel("DIRAC")

    def test_infer_parallel(self):
        self.method_tests_parallel("INFER")

    def test_race_parallel(self):
        self.method_tests_parallel("RACE", iterations=100)

    def test_crane_parallel(self):
        self.method_tests_parallel("CRANE")


if __name__ == "__main__":
    unittest.main()
