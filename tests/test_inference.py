"""Small exact oracles, invariants, and independent statistical cross-checks."""
import itertools
import math
import unittest
import numpy as np
from scipy import stats
from scripts.distances import l2_squared,l2_grand,integration_cholesky
from scripts.permutation_tests import (permutation_test,partition_indices,allocation_count,
                                       independent_distribution,inclusive_pvalue)
from scripts.posthoc_tests import holm_adjust,pairwise_posthoc
from scripts.reference_tests import anova_group_effect,paired_condition_effect
from scripts.metrics import binomial_interval


class DistanceTests(unittest.TestCase):
    def test_constant_distance(self):
        self.assertAlmostEqual(l2_squared([3,3],[0,0],[0,1]),9)

    def test_linear_distance(self):
        self.assertAlmostEqual(l2_squared([0,0.4,1],[0,0,0],[0,0.4,1]),1/3)

    def test_grand_sum_is_not_squared_again(self):
        means=np.array([[0,0],[1,1],[3,3]])
        self.assertAlmostEqual(l2_grand(means,[0,1]),14)

    def test_common_grid_refinement(self):
        self.assertAlmostEqual(l2_squared([1,5],[0,0],[0,2]),
                               l2_squared([1,2,5],[0,0,0],[0,0.5,2]))

    def test_mass_matrix(self):
        x=np.array([0,0.1,0.7,2.0]);rng=np.random.default_rng(12)
        for _ in range(20):
            d=rng.normal(size=4)
            self.assertAlmostEqual(np.sum((d@integration_cholesky(x))**2),l2_squared(d,d*0,x),places=12)

    def test_invalid_grids(self):
        for x in ([0,0],[1,0],[0,np.nan]):
            with self.assertRaises(ValueError):integration_cholesky(x)


class PermutationTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([0,0.2,1.0]);self.rng=np.random.default_rng(321)
        self.groups=(self.rng.normal(size=(2,3)),self.rng.normal(size=(3,3)),self.rng.normal(size=(2,3)))

    def oracle(self,groups,design='independent'):
        arrays=tuple(groups);pool=np.concatenate(arrays);offsets=np.cumsum([0]+[len(a) for a in arrays])
        indices=tuple(np.arange(offsets[i],offsets[i+1]) for i in range(len(arrays)))
        def statistic(*index_groups):
            return l2_grand([pool[np.asarray(i,dtype=int)].mean(0) for i in index_groups],self.x)
        return stats.permutation_test(indices,statistic,permutation_type='samples' if design=='paired' else 'independent',
                                      n_resamples=np.inf,alternative='greater',vectorized=False,random_state=3)

    def test_partition_count_unequal_sizes(self):
        self.assertEqual(allocation_count((2,3,2)),210)
        rows=partition_indices((2,3,2))
        self.assertEqual(len(rows),210)
        self.assertEqual(len(set(map(tuple,rows))),210)
        for row in rows:self.assertEqual(sorted(row),list(range(7)))

    def test_exact_two_group_against_scipy(self):
        g=self.groups[:2];r=permutation_test(g,self.x,exact='always',return_distribution=True)
        ref=self.oracle(g)
        self.assertAlmostEqual(r.statistic,ref.statistic,places=12)
        self.assertAlmostEqual(r.pvalue,ref.pvalue,places=12)
        np.testing.assert_allclose(np.sort(r.null_distribution),np.sort(ref.null_distribution),rtol=1e-12,atol=1e-14)

    def test_exact_three_group_against_scipy(self):
        r=permutation_test(self.groups,self.x,exact='always',return_distribution=True)
        ref=self.oracle(self.groups)
        self.assertAlmostEqual(r.statistic,ref.statistic,places=12)
        self.assertAlmostEqual(r.pvalue,ref.pvalue,places=12)
        np.testing.assert_allclose(np.sort(r.null_distribution),np.sort(ref.null_distribution),rtol=1e-12,atol=1e-14)

    def test_exact_paired_against_scipy(self):
        g=(self.groups[0],self.groups[2]);r=permutation_test(g,self.x,design='paired',exact='always',return_distribution=True)
        ref=self.oracle(g,'paired')
        self.assertEqual(r.total_allocations,4)
        self.assertAlmostEqual(r.pvalue,ref.pvalue)
        np.testing.assert_allclose(np.sort(r.null_distribution),np.sort(ref.null_distribution),rtol=1e-12)

    def test_paired_not_independent_allocation_space(self):
        a=self.rng.normal(size=(4,3));b=self.rng.normal(size=(4,3))
        p=permutation_test((a,b),self.x,design='paired',exact='always')
        i=permutation_test((a,b),self.x,exact='always')
        self.assertEqual(p.evaluated_permutations,16);self.assertEqual(i.evaluated_permutations,70)

    def test_all_zero_statistics(self):
        groups=[np.zeros((2,3)) for _ in range(3)]
        for exact in ('always','never'):
            self.assertEqual(permutation_test(groups,self.x,permutations=7,exact=exact).pvalue,1)
            self.assertEqual(permutation_test(groups[:2],self.x,design='paired',permutations=7,exact=exact).pvalue,1)

    def test_monte_carlo_plus_one(self):
        self.assertEqual(inclusive_pvalue([0,1,2],5),1/4)
        self.assertEqual(inclusive_pvalue([0,1,2],2),2/4)
        self.assertEqual(inclusive_pvalue([0,1,2],2,True),1/3)

    def test_seed_reproducible(self):
        a=permutation_test(self.groups,self.x,exact='never',permutations=71,rng=np.random.default_rng(9),return_distribution=True)
        b=permutation_test(self.groups,self.x,exact='never',permutations=71,rng=np.random.default_rng(9),return_distribution=True)
        np.testing.assert_array_equal(a.null_distribution,b.null_distribution)

    def test_group_order_invariance_exact(self):
        a=permutation_test(self.groups,self.x,exact='always')
        b=permutation_test(tuple(reversed(self.groups)),self.x,exact='always')
        self.assertAlmostEqual(a.statistic,b.statistic)
        self.assertAlmostEqual(a.pvalue,b.pvalue)

    def test_scaling_and_common_shift(self):
        a=permutation_test(self.groups,self.x,exact='always')
        for scale,shift in [(1e-6,0),(1e6,0),(1,37)]:
            b=permutation_test(tuple(g*scale+shift for g in self.groups),self.x,exact='always')
            self.assertAlmostEqual(b.statistic/scale**2,a.statistic,places=10)
            self.assertEqual(a.pvalue,b.pvalue)

    def test_grand_for_two_groups(self):
        r=permutation_test(self.groups[:2],self.x,exact='always')
        self.assertAlmostEqual(r.statistic,l2_squared(self.groups[0].mean(0),self.groups[1].mean(0),self.x))

    def test_exact_four_group_generalization(self):
        groups=tuple(self.rng.normal(size=(2,3)) for _ in range(4))
        r=permutation_test(groups,self.x,exact='always')
        ref=self.oracle(groups)
        self.assertEqual(r.evaluated_permutations,2520)
        self.assertAlmostEqual(r.pvalue,ref.pvalue)
        self.assertAlmostEqual(r.statistic,ref.statistic)

    def test_exact_safety_limit(self):
        a=np.zeros((10,3))
        with self.assertRaises(ValueError):permutation_test((a,a),self.x,exact='always')

    def test_invalid_paired_shape(self):
        with self.assertRaises(ValueError):permutation_test(self.groups[:2],self.x,design='paired')

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):permutation_test(self.groups,self.x,permutations=0)


class PosthocAndReferenceTests(unittest.TestCase):
    def test_holm_known_values(self):
        np.testing.assert_allclose(holm_adjust([0.01,0.04,0.03]),[0.03,0.06,0.06])

    def test_holm_order_and_ties(self):
        p=np.array([0.02,0.02,0.8,0.01]);order=[3,1,0,2]
        np.testing.assert_allclose(holm_adjust(p)[order],holm_adjust(p[order]))
        self.assertTrue(np.all(holm_adjust(p)>=p))

    def test_holm_invalid(self):
        for p in ([np.nan],[],[-0.1],[1.1]):
            with self.assertRaises(ValueError):holm_adjust(p)

    def test_pair_specific_null_and_full_family(self):
        x=np.array([0,1]);g=[np.array([[0,0],[1,1]]),np.array([[2,2],[3,3]]),np.array([[10,10],[11,11]])]
        rows=pairwise_posthoc(g,x,exact='always',omnibus_pvalue=0.01)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r['family_size']==3 and r['allocations_evaluated']==6 for r in rows))
        direct=permutation_test(g[:2],x,exact='always')
        self.assertEqual(rows[0]['p_raw'],direct.pvalue)
        # The third group's large effect does not enter this pair's null.
        modified=pairwise_posthoc([g[0],g[1],g[2]*100],x,exact='always',omnibus_pvalue=0.01)
        self.assertEqual(rows[0]['p_raw'],modified[0]['p_raw'])

    def test_omnibus_gate(self):
        x=np.array([0,1]);groups=[np.zeros((3,2)),np.ones((3,2))*10,np.ones((3,2))*20]
        rows=pairwise_posthoc(groups,x,exact='always',omnibus_pvalue=1,alpha=0.5)
        self.assertTrue(any(r['reject_holm'] for r in rows))
        self.assertFalse(any(r['reject_after_gate'] for r in rows))

    def test_independent_anova_matches_scipy(self):
        rng=np.random.default_rng(12);g=[rng.normal(size=(n,5)) for n in (3,4,6)]
        ref=stats.f_oneway(*[a.mean(1) for a in g])
        np.testing.assert_allclose(anova_group_effect(*g),[ref.statistic,ref.pvalue],rtol=1e-12)

    def test_paired_reference_matches_t(self):
        rng=np.random.default_rng(12);a,b=rng.normal(size=(2,7,4))
        ref=stats.ttest_rel(a.mean(1),b.mean(1))
        np.testing.assert_allclose(paired_condition_effect(a,b),[ref.statistic**2,ref.pvalue],rtol=1e-12)

    def test_binomial_endpoints(self):
        self.assertEqual(binomial_interval(0,25)[0],0)
        self.assertEqual(binomial_interval(25,25)[1],1)
        self.assertGreater(binomial_interval(0,25)[1],0)
        self.assertLess(binomial_interval(25,25)[0],1)


if __name__=='__main__':unittest.main()
