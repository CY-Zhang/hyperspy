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
    @pytest.fixture
    def sparse_array(self):
        data = np.ones(shape=(10, 10, 1, 1))
        data[::2,::2] = -1
        data = data * np.random.random((1, 1, 3, 4))
        s = signals.BaseSignal(data=data)
        s = s.transpose(signal_axes=(0,1))
        return s

    def test_svd(self,sparse_array):
        sparse_array.tensor_decomposition(rank=(2, 2, 12))


