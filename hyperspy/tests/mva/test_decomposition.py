# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pytest

from hyperspy import signals
from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed
from hyperspy.decorators import lazifyTestClass
from tempfile import TemporaryDirectory
from os.path import join


class TestNdAxes:

    def setup_method(self, method):
        # Create three signals with dimensions:
        # s1 : <BaseSignal, title: , dimensions: (4, 3, 2|2, 3)>
        # s2 : <BaseSignal, title: , dimensions: (2, 3|4, 3, 2)>
        # s12 : <BaseSignal, title: , dimensions: (2, 3|4, 3, 2)>
        # Where s12 data is transposed in respect to s2
        dc1 = np.random.random((2, 3, 4, 3, 2))
        dc2 = np.rollaxis(np.rollaxis(dc1, -1), -1)
        s1 = signals.BaseSignal(dc1.copy())
        s2 = signals.BaseSignal(dc2)
        s12 = signals.BaseSignal(dc1.copy())
        for i, axis in enumerate(s1.axes_manager._axes):
            if i < 3:
                axis.navigate = True
            else:
                axis.navigate = False
        for i, axis in enumerate(s2.axes_manager._axes):
            if i < 2:
                axis.navigate = True
            else:
                axis.navigate = False
        for i, axis in enumerate(s12.axes_manager._axes):
            if i < 3:
                axis.navigate = False
            else:
                axis.navigate = True
        self.s1 = s1
        self.s2 = s2
        self.s12 = s12

    def test_consistency(self):
        s1 = self.s1
        s2 = self.s2
        s12 = self.s12
        s1.decomposition()
        s2.decomposition()
        s12.decomposition()
        np.testing.assert_array_almost_equal(s2.learning_results.loadings,
                                             s12.learning_results.loadings)
        np.testing.assert_array_almost_equal(s2.learning_results.factors,
                                             s12.learning_results.factors)
        np.testing.assert_array_almost_equal(s1.learning_results.loadings,
                                             s2.learning_results.factors)
        np.testing.assert_array_almost_equal(s1.learning_results.factors,
                                             s2.learning_results.loadings)

    def test_consistency_poissonian(self):
        s1 = self.s1
        s1n000 = self.s1.inav[0, 0, 0]
        s2 = self.s2
        s12 = self.s12
        s1.decomposition(normalize_poissonian_noise=True)
        s2.decomposition(normalize_poissonian_noise=True)
        s12.decomposition(normalize_poissonian_noise=True)
        np.testing.assert_array_almost_equal(s2.learning_results.loadings,
                                             s12.learning_results.loadings)
        np.testing.assert_array_almost_equal(s2.learning_results.factors,
                                             s12.learning_results.factors)
        np.testing.assert_array_almost_equal(s1.learning_results.loadings,
                                             s2.learning_results.factors)
        np.testing.assert_array_almost_equal(s1.learning_results.factors,
                                             s2.learning_results.loadings)
        # Check that views of the data don't change. See #871
        np.testing.assert_array_equal(s1.inav[0, 0, 0].data, s1n000.data)


@lazifyTestClass
class TestGetModel:
    def setup_method(self, method):
        np.random.seed(100)
        sources = signals.Signal1D(np.random.standard_t(.5, size=(3, 100)))
        np.random.seed(100)
        maps = signals.Signal2D(np.random.standard_t(.5, size=(3, 4, 5)))
        self.s = (sources.inav[0] * maps.inav[0].T
                  + sources.inav[1] * maps.inav[1].T
                  + sources.inav[2] * maps.inav[2].T)

    def test_get_decomposition_model(self):
        s = self.s
        s.decomposition(algorithm='svd')
        sc = self.s.get_decomposition_model(3)
        rms = np.sqrt(((sc.data - s.data)**2).sum())
        assert rms < 5e-7

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_get_bss_model(self):
        s = self.s
        s.decomposition(algorithm='svd')
        s.blind_source_separation(3)
        sc = self.s.get_bss_model()
        rms = np.sqrt(((sc.data - s.data)**2).sum())
        assert rms < 5e-7


@lazifyTestClass
class TestGetExplainedVarinaceRatio:

    def setup_method(self, method):
        s = signals.BaseSignal(np.empty(1))
        self.s = s

    def test_data(self):
        self.s.learning_results.explained_variance_ratio = np.asarray([2, 4])
        np.testing.assert_array_equal(
            self.s.get_explained_variance_ratio().data,
            np.asarray([2, 4]))

    def test_no_evr(self):
        with pytest.raises(AttributeError):
            self.s.get_explained_variance_ratio()


class TestEstimateElbowPosition:

    def setup_method(self, method):
        s = signals.BaseSignal(np.empty(1))
        s.learning_results.explained_variance_ratio = \
            np.asarray([10e-1, 5e-2, 9e-3, 1e-3, 9e-5, 5e-5, 3.0e-5,
                        2.2e-5, 1.9e-5, 1.8e-5, 1.7e-5, 1.6e-5])
        self.s = s

    def test_elbow_position(self):
        variance = self.s.learning_results.explained_variance_ratio
        elbow = self.s._estimate_elbow_position(variance)
        assert elbow == 4

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_store_number_significant_components(self):
        np.random.seed(1)
        s = signals.Signal1D(np.random.random((20, 100)))
        s.decomposition()
        assert s.learning_results.number_significant_components == 2
        # Check that number_significant_components is reset properly
        s.decomposition(algorithm='nmf', output_dimension=2)
        assert s.learning_results.number_significant_components is None


class TestReverseDecompositionComponent:

    def setup_method(self, method):
        s = signals.BaseSignal(np.zeros(1))
        self.factors = np.ones([2, 3])
        self.loadings = np.ones([2, 3])
        s.learning_results.factors = self.factors.copy()
        s.learning_results.loadings = self.loadings.copy()
        self.s = s

    def test_reversal_factors_one_component_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(self.s.learning_results.factors[:, 0],
                                      self.factors[:, 0] * -1)

    def test_reversal_loadings_one_component_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(self.s.learning_results.loadings[:, 0],
                                      self.loadings[:, 0] * -1)

    def test_reversal_factors_one_component_not_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(self.s.learning_results.factors[:, 1:],
                                      self.factors[:, 1:])

    def test_reversal_loadings_one_component_not_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(self.s.learning_results.loadings[:, 1:],
                                      self.loadings[:, 1:])

    def test_reversal_factors_multiple_components_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(self.s.learning_results.factors[:, (0, 2)],
                                      self.factors[:, (0, 2)] * -1)

    def test_reversal_loadings_multiple_components_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(self.s.learning_results.loadings[:, (0, 2)],
                                      self.loadings[:, (0, 2)] * -1)

    def test_reversal_factors_multiple_components_not_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(self.s.learning_results.factors[:, 1],
                                      self.factors[:, 1])

    def test_reversal_loadings_multiple_components_not_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(self.s.learning_results.loadings[:, 1],
                                      self.loadings[:, 1])


class TestNormalizeComponents():

    def setup_method(self, method):
        s = signals.BaseSignal(np.zeros(1))
        self.factors = np.ones([2, 3])
        self.loadings = np.ones([2, 3])
        s.learning_results.factors = self.factors.copy()
        s.learning_results.loadings = self.loadings.copy()
        s.learning_results.bss_factors = self.factors.copy()
        s.learning_results.bss_loadings = self.loadings.copy()
        self.s = s

    def test_normalize_bss_factors(self):
        s = self.s
        s.normalize_bss_components(target="factors",
                                   function=np.sum)
        np.testing.assert_array_equal(s.learning_results.bss_factors,
                                      self.factors / 2.)
        np.testing.assert_array_equal(s.learning_results.bss_loadings,
                                      self.loadings * 2.)

    def test_normalize_bss_loadings(self):
        s = self.s
        s.normalize_bss_components(target="loadings",
                                   function=np.sum)
        np.testing.assert_array_equal(s.learning_results.bss_factors,
                                      self.factors * 2.)
        np.testing.assert_array_equal(s.learning_results.bss_loadings,
                                      self.loadings / 2.)

    def test_normalize_decomposition_factors(self):
        s = self.s
        s.normalize_decomposition_components(target="factors",
                                             function=np.sum)
        np.testing.assert_array_equal(s.learning_results.factors,
                                      self.factors / 2.)
        np.testing.assert_array_equal(s.learning_results.loadings,
                                      self.loadings * 2.)

    def test_normalize_decomposition_loadings(self):
        s = self.s
        s.normalize_decomposition_components(target="loadings",
                                             function=np.sum)
        np.testing.assert_array_equal(s.learning_results.factors,
                                      self.factors * 2.)
        np.testing.assert_array_equal(s.learning_results.loadings,
                                      self.loadings / 2.)


class TestPrintInfo:

    def setup_method(self, method):
        self.s = signals.Signal1D(np.random.random((20, 100)))

    @pytest.mark.parametrize("algorithm", ["svd", "fast_svd", "mlpca", "fast_mlpca"])
    def test_decomposition(self, algorithm, capfd):
        self.s.decomposition(algorithm=algorithm, output_dimension=2)
        captured = capfd.readouterr()
        assert "Decomposition info:" in captured.out

    # Warning filter can be removed after scikit-learn >= 0.22
    # See sklearn.decomposition.sparse_pca.SparsePCA docstring
    @pytest.mark.filterwarnings("ignore:normalize_components=False:DeprecationWarning")
    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_decomposition_sklearn(self, capfd):
        for algorithm in ["sklearn_pca", "nmf",
                          "sparse_pca", "mini_batch_sparse_pca"]:
            print(algorithm)
            self.s.decomposition(algorithm=algorithm, output_dimension=2)
            captured = capfd.readouterr()
            assert "Decomposition info:" in captured.out
            assert "scikit-learn estimator:" in captured.out

    def test_decomposition_no_print(self, capfd):
        self.s.decomposition(algorithm="svd", output_dimension=2,
                             print_info=False)
        captured = capfd.readouterr()
        assert "Decomposition info:" not in captured.out


class TestReturnInfo:

    def setup_method(self, method):
        self.s = signals.Signal1D(np.random.random((20, 100)))

    @pytest.mark.parametrize("algorithm", ["svd", "fast_svd", "mlpca", "fast_mlpca"])
    def test_decomposition_not_supported(self, algorithm):
        assert self.s.decomposition(algorithm=algorithm, return_info=True,
                                    output_dimension=2) is None

    # Warning filter can be removed after scikit-learn >= 0.22
    # See sklearn.decomposition.sparse_pca.SparsePCA docstring
    @pytest.mark.filterwarnings("ignore:normalize_components=False:DeprecationWarning")
    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_decomposition_supported_return_true(self):
        for algorithm in ["RPCA_GoDec", "ORPCA", "ORNMF"]:
            assert self.s.decomposition(
                algorithm=algorithm,
                return_info=True,
                output_dimension=2) is not None
        for algorithm in ["sklearn_pca", "nmf",
                          "sparse_pca", "mini_batch_sparse_pca"]:
            assert self.s.decomposition(
                algorithm=algorithm,
                return_info=True,
                output_dimension=2) is not None

    # Warning filter can be removed after scikit-learn >= 0.22
    # See sklearn.decomposition.sparse_pca.SparsePCA docstring
    @pytest.mark.filterwarnings("ignore:normalize_components=False:DeprecationWarning")
    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_decomposition_supported_return_false(self):
        for algorithm in ["RPCA_GoDec", "ORPCA", "ORNMF"]:
            assert self.s.decomposition(
                algorithm=algorithm,
                return_info=False,
                output_dimension=2) is None
        for algorithm in ["sklearn_pca", "nmf",
                          "sparse_pca", "mini_batch_sparse_pca"]:
            assert self.s.decomposition(
                algorithm=algorithm,
                return_info=False,
                output_dimension=2) is None

    # Warning filter can be removed after scikit-learn >= 0.22
    # See sklearn.decomposition.sparse_pca.SparsePCA docstring
    @pytest.mark.filterwarnings("ignore:normalize_components=False:DeprecationWarning")
    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    @pytest.mark.parametrize("algorithm", ["RPCA_GoDec", "ORPCA", "ORNMF"])
    def test_decomposition_output_dimension_not_given(self, algorithm):
        with pytest.raises(ValueError, match="the output_dimension must be specified"):
            self.s.decomposition(algorithm=algorithm, return_info=False)


class TestNonFloatTypeError:

    def setup_method(self, method):
        self.s_int = signals.Signal1D(
            (np.random.random((20, 100)) * 20).astype('int'))
        self.s_float = signals.Signal1D(np.random.random((20, 100)))

    def test_decomposition_error(self):
        self.s_float.decomposition()
        with pytest.raises(TypeError):
            self.s_int.decomposition()


class TestLoadDecompositionResults:

    def setup_method(self, method):
        self.s = signals.Signal1D([[1.1, 1.2, 1.4, 1.3], [1.5, 1.5, 1.4, 1.2]])

    def test_load_decomposition_results(self):
        # Test whether the sequence of loading learning results and then
        # saving the signal causes errors. See #2093.
        with TemporaryDirectory() as tmpdir:
            self.s.decomposition()
            fname1 = join(tmpdir, "results.npz")
            self.s.learning_results.save(fname1)
            self.s.learning_results.load(fname1)
            fname2 = join(tmpdir, "output.hspy")
            self.s.save(fname2)
            assert isinstance(
                self.s.learning_results.decomposition_algorithm, str)

class TestComplexSignalDecomposition:

    def setup_method(self, method):
        real = np.random.random((8,8))
        imag = np.random.random((8,8))
        s_complex_dtype = signals.ComplexSignal1D(real + 1j*imag - 1j*imag)
        s_real_dtype = signals.ComplexSignal1D(real)
        s_complex_dtype.decomposition()
        s_real_dtype.decomposition()
        self.s_complex_dtype = s_complex_dtype
        self.s_real_dtype = s_real_dtype

    def test_imaginary_is_zero(self):
        np.testing.assert_allclose(self.s_complex_dtype.data.imag, 0.0)

    def test_decomposition_independent_of_complex_dtype(self):
        complex_pca = self.s_complex_dtype.get_decomposition_model(5)
        real_pca = self.s_real_dtype.get_decomposition_model(5)
        np.testing.assert_almost_equal((complex_pca - real_pca).data.max(), 0.0)

    def test_first_r_values_of_scree_non_zero(self):
        """For low-rank matrix by creating a = RandomComplex(m, r) and
        b = RandomComplex(n, r), then performing PCA on the result of a.b^T
        (i.e. an m x n matrix).
        The first r values of scree plot / singular values should be non-zero.
        """
        m, n, r = 32, 32, 3

        A = (np.random.random((m,r)) + 1j* np.random.random((m,r)))
        B = (np.random.random((n,r)) + 1j* np.random.random((n,r)))

        s = signals.ComplexSignal1D(np.dot(A,B.T))
        s.decomposition()
        np.testing.assert_almost_equal(
            s.get_explained_variance_ratio().data[r:].sum(), 0)
