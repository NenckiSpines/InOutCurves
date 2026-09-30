import io
import os
import pickle
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.subject_result import SubjectResult
from scripts.subject_results import SubjectResults
from scripts.cache import NumericUnpickler,save_cache,read_cache
from scripts.simulation import load_source,calibrate,simulate_experiment
from scripts.compare_groups import compare_dataset
from scripts.paths import ROOT,INPUT_DATA
from scripts.sigmoid_fit import SigmoidFitter


class SubjectDataTests(unittest.TestCase):
    def data(self):
        return SubjectResults(np.array([0.,1.]),[
            SubjectResult('s2',[0,4],'POST'),SubjectResult('s1',[0,1],'PRE'),
            SubjectResult('s2',[0,2],'PRE'),SubjectResult('s1',[0,3],'POST'),
            SubjectResult('s1',[0,5],'POST')])

    def test_average_within_subject_condition(self):
        arrays,ids=self.data().analysis_arrays(['PRE','POST'],'paired')
        self.assertEqual(ids,(['s1','s2'],['s1','s2']))
        np.testing.assert_array_equal(arrays[1],[[0,4],[0,4]])
        self.assertEqual(self.data().group_by_group(['PRE','POST']).means[1,1],4)

    def test_missing_pairs_error(self):
        d=self.data();d.results=[SubjectResult('s3' if (r.subject_id=='s2' and r.group_id=='POST') else r.subject_id,r.responses,r.group_id) for r in d.results]
        with self.assertRaisesRegex(ValueError,'same subject IDs'):d.analysis_arrays(['PRE','POST'],'paired')

    def test_independent_shared_ids_error(self):
        with self.assertRaisesRegex(ValueError,'share subject IDs'):self.data().analysis_arrays(['PRE','POST'],'independent')

    def test_paired_order_invariance(self):
        d=self.data();a=compare_dataset(d,['PRE','POST'],design='paired',exact='always')
        d.results=list(reversed(d.results));b=compare_dataset(d,['PRE','POST'],design='paired',exact='always')
        self.assertEqual(a['omnibus']['pvalue'],b['omnibus']['pvalue'])
        self.assertEqual(a['omnibus']['statistic'],b['omnibus']['statistic'])

    def test_pair_reassignment_preserves_ids(self):
        d=self.data().group_by_subject();r=d.randomize_groups(design='paired',groups=['PRE','POST'],rng=np.random.default_rng(3))
        for sid in ('s1','s2'):
            original=sorted(tuple(x.responses) for x in d.results if x.subject_id==sid)
            new=sorted(tuple(x.responses) for x in r.results if x.subject_id==sid)
            self.assertEqual(original,new)

    def test_independent_group_sizes_preserved(self):
        d=SubjectResults.from_csv(INPUT_DATA/'stress_response_ca1.csv')
        r=d.randomize_groups(groups=['CTR','ANH','RES'],rng=np.random.default_rng(4))
        arrays,ids=r.analysis_arrays(['CTR','ANH','RES'],'independent')
        self.assertEqual(tuple(map(len,arrays)),(6,6,5))
        self.assertEqual(len(set(sum([list(v) for v in ids],[]))),17)

    def test_roundtrip_csv(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'curves.csv';data=self.data();data.to_csv(path)
            loaded=SubjectResults.from_csv(path)
            self.assertEqual(len(data.results),len(loaded.results))
            for a,b in zip(data.results,loaded.results):
                self.assertEqual((a.group_id,a.subject_id),(b.group_id,b.subject_id))
                np.testing.assert_array_equal(a.responses,b.responses)

    def test_missing_responses_no_interpolation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'missing.csv';path.write_text('GROUP,SUBJECT,0,1\nA,1,0,\n')
            with self.assertRaises(ValueError):SubjectResults.from_csv(path)

    def test_archive_data_available_and_paired(self):
        data=SubjectResults.from_csv(INPUT_DATA/'ltp_paired.csv')
        arrays,ids=data.analysis_arrays(['PRE','POST'],'paired')
        self.assertEqual([len(a) for a in arrays],[24,24])
        self.assertEqual(ids[0],ids[1])

    def test_archived_two_group_exact_reference(self):
        data=SubjectResults.from_csv(INPUT_DATA/'stress_response_ca1.csv')
        report=compare_dataset(data,['ANH','RES'],exact='always')
        self.assertEqual(report['omnibus']['evaluated_permutations'],462)
        self.assertAlmostEqual(report['omnibus']['pvalue'],107/462)

    def test_unknown_group_rejected(self):
        with self.assertRaises(ValueError):self.data().subject_arrays(['PRE','missing'])


class CacheAndSimulationTests(unittest.TestCase):
    def test_cache_roundtrip_and_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'numbers.pkl'
            save_cache(path,{'seed':1},{'values':np.array([1,2]),'ok':True})
            self.assertEqual(read_cache(path,{'seed':1})['data']['values'],[1,2])
            with self.assertRaises(ValueError):read_cache(path,{'seed':2})
            self.assertIsNone(read_cache(path,{'seed':2},True))

    def test_unsafe_pickle_global_rejected(self):
        with self.assertRaises(pickle.UnpicklingError):NumericUnpickler(io.BytesIO(pickle.dumps(os.system))).load()

    def test_numeric_numpy_legacy_pickle(self):
        p=pickle.dumps([np.float64(0.2),np.float64(0.3)],protocol=4)
        np.testing.assert_array_equal(NumericUnpickler(io.BytesIO(p)).load(),[0.2,0.3])

    def test_simulation_models_differ_explicitly(self):
        source=load_source(INPUT_DATA/'ltp_paired.csv','PRE','POST')
        new=calibrate(source,'within-group',ROOT);old=calibrate(source,'archive',ROOT)
        self.assertEqual(new.recordings_per_subject,1);self.assertEqual(old.recordings_per_subject,3)
        self.assertAlmostEqual(new.gain_location+new.gain_scale,1)
        self.assertAlmostEqual(old.gain_location+old.gain_scale,1)
        self.assertFalse(np.allclose(new.noise_sd,old.noise_sd))

    def test_null_simulation_independent_not_copied(self):
        source=load_source(INPUT_DATA/'ltp_paired.csv','PRE','POST');model=calibrate(source,'within-group',ROOT)
        a,b=simulate_experiment(model,5,'null',np.random.default_rng(9))
        self.assertEqual(a.shape,(5,13));self.assertFalse(np.array_equal(a,b))

    def test_simulation_deterministic(self):
        source=load_source(INPUT_DATA/'ltp_paired.csv','PRE','POST');model=calibrate(source,'within-group',ROOT)
        a=simulate_experiment(model,5,'alternative',np.random.default_rng(4))
        b=simulate_experiment(model,5,'alternative',np.random.default_rng(4))
        np.testing.assert_array_equal(a,b)

    def test_sigmoid_helper_no_global_variables(self):
        x=np.linspace(0,300,13);truth=[1,140,0.04,0.02]
        y=SigmoidFitter.sigmoid(x,*truth);fit=SigmoidFitter(x,y)
        fit.fit_sigmoid(p0=[0.9,130,0.03,0])
        np.testing.assert_allclose(fit.y_fitted,y,atol=1e-6)
        self.assertGreater(fit.r_squared(),0.999999)


if __name__=='__main__':unittest.main()
