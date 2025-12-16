# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.optimize.experimental.optimizer.global_best_custom import CustomGlobalBestPSO


class TestCustomGlobalBestPSO(unittest.TestCase):
    def setUp(self):
        self.n_particles = 3
        self.dimensions = 2
        self.bounds = ([-5, -5], [5, 5])
        self.options = {
            'c1': 0.5,
            'c2': 0.3,
            'w': 0.9,
        }

    def test_init_without_breakpoints(self):
        """Test initialization without breakpoints"""
        optimizer = CustomGlobalBestPSO(n_particles=self.n_particles,
                                      dimensions=self.dimensions,
                                      options=self.options,
                                      bounds=self.bounds)
        self.assertIsNone(optimizer.breakpoint_cost)
        self.assertIsNone(optimizer.breakpoint_pos)

    @patch('experimental.optimizer.global_best_custom.compute_pbest')
    def test_init_with_breakpoints(self, mock_compute_pbest):
        """Test initialization with breakpoints"""
        breakpoint_pos = [[1, 1], [2, 2], [3, 3]]
        breakpoint_cost = [0.1, 0.2, 0.3]
        mock_compute_pbest.return_value = (np.array([[1, 1], [2, 2], [3, 3]]), np.array([0.1, 0.2, 0.3]))
        
        optimizer = CustomGlobalBestPSO(n_particles=self.n_particles,
                                      dimensions=self.dimensions,
                                      options=self.options,
                                      bounds=self.bounds,
                                      breakpoint_pos=breakpoint_pos,
                                      breakpoint_cost=breakpoint_cost)
        
        self.assertEqual(optimizer.breakpoint_cost, breakpoint_cost)
        self.assertEqual(optimizer.breakpoint_pos, breakpoint_pos)

    @patch('experimental.optimizer.global_best_custom.compute_pbest')
    def test_computer_next_pos_exact_particles(self, mock_compute_pbest):
        """Test computer_next_pos with exact number of particles"""
        breakpoint_pos = [[1, 1], [2, 2], [3, 3]]
        breakpoint_cost = [0.1, 0.2, 0.3]
        mock_compute_pbest.return_value = (np.array([[1, 1], [2, 2], [3, 3]]), np.array([0.1, 0.2, 0.3]))
        
        optimizer = CustomGlobalBestPSO(n_particles=3,
                                      dimensions=2,
                                      options=self.options,
                                      bounds=self.bounds,
                                      breakpoint_pos=breakpoint_pos,
                                      breakpoint_cost=breakpoint_cost)
        
        optimizer.computer_next_pos()
        self.assertEqual(optimizer.swarm.position.shape, (3, 2))

    @patch('experimental.optimizer.global_best_custom.compute_pbest')
    def test_computer_next_pos_partial_particles(self, mock_compute_pbest):
        """Test computer_next_pos with partial particles"""
        breakpoint_pos = [[1, 1], [2, 2], [3, 3], [4, 4]]
        breakpoint_cost = [0.1, 0.2, 0.3, 0.4]
        mock_compute_pbest.return_value = (np.array([[1, 1], [2, 2], [3, 3]]), np.array([0.1, 0.2, 0.3]))
        
        optimizer = CustomGlobalBestPSO(n_particles=3,
                                      dimensions=2,
                                      options=self.options,
                                      bounds=self.bounds,
                                      breakpoint_pos=breakpoint_pos,
                                      breakpoint_cost=breakpoint_cost)
        
        optimizer.computer_next_pos()
        self.assertEqual(optimizer.swarm.position.shape, (3, 2))

    @patch('experimental.optimizer.global_best_custom.compute_pbest')
    def test_computer_next_pos_empty_current_cost(self, mock_compute_pbest):
        """Test computer_next_pos with empty current_cost"""
        breakpoint_pos = [[1, 1], [2, 2]]
        breakpoint_cost = [0.1, 0.2]
        mock_compute_pbest.return_value = (np.array([[1, 1], [2, 2], [3, 3]]), np.array([0.1, 0.2, 0.3]))
        
        optimizer = CustomGlobalBestPSO(n_particles=3,
                                      dimensions=2,
                                      options=self.options,
                                      bounds=self.bounds,
                                      breakpoint_pos=breakpoint_pos,
                                      breakpoint_cost=breakpoint_cost)
        
        optimizer.swarm.current_cost = np.array([])
        optimizer.computer_next_pos()
        self.assertEqual(optimizer.swarm.position.shape, (3, 2))
