#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim)
# Title: env.py
# Explanation: Generates Mixed-Level problems (50% Identity / 50% Diversity) across ALL phases.
#              PE signals are generated only in Test phases for the Oddball condition.
##### DATA FILES ARE UNAVAILABLE UNTIL THE PAPER PUBLISHED #####

import numpy as np
import random
import itertools
from typing import Tuple

class OddOneOutEnv:
    def __init__(self):
        self.n_attrs = 4
        self.n_vals = 3
        self.n_cards = 4
        self.action_space = list(range(self.n_cards))
        self.triplet_indices = list(itertools.combinations(range(self.n_cards), 3))

    def reset(self, phase: str = 'practice', condition: str = 'standard') -> Tuple[np.ndarray, float, int, str]:
        """
        Resets the environment.
        Args:
            phase: 'practice', 'block1', or 'block2'.
            condition: 'standard' or 'oddball'.
        """
        trial_type = np.random.choice(['identity', 'diversity'], p=[0.5, 0.5])

        observation, target_idx = self._generate_valid_trial(trial_type)

        pe_signal = 0.0
        if phase in ['block1', 'block2'] and condition == 'oddball':
            if random.random() < 0.5:
                pe_signal = 1.0

        return observation, pe_signal, target_idx, trial_type

    def step(self, action: int, target_idx: int) -> Tuple[float, bool, dict]:
        reward = 1.0 if action == target_idx else 0.0
        done = True
        return reward, done, {}

    def _generate_valid_trial(self, trial_type: str) -> Tuple[np.ndarray, int]:
        while True:
            set_cards = self._create_set_by_type(trial_type)
            distractor = np.random.randint(0, self.n_vals, size=(1, self.n_attrs))
            cards = np.vstack([set_cards, distractor])

            if len(np.unique(cards, axis=0)) < self.n_cards:
                continue

            valid_triplets_count = 0
            valid_triplet_indices = None

            for indices in self.triplet_indices:
                triplet = cards[list(indices)]
                if self._is_set(triplet):
                    valid_triplets_count += 1
                    valid_triplet_indices = indices

            if valid_triplets_count == 1:
                all_indices = set(range(self.n_cards))
                set_indices = set(valid_triplet_indices)
                target_idx = list(all_indices - set_indices)[0]
                perm = np.random.permutation(self.n_cards)
                cards = cards[perm]
                new_target_idx = np.where(perm == target_idx)[0][0]
                return cards, new_target_idx

    def _create_set_by_type(self, trial_type: str) -> np.ndarray:
        cards = np.zeros((3, self.n_attrs), dtype=int)
        if trial_type == 'identity':
            attr_types = ['same', 'same', 'same', 'diff']
        else:
            attr_types = ['same', 'diff', 'diff', 'diff']
        random.shuffle(attr_types)

        for i, attr_type in enumerate(attr_types):
            if attr_type == 'same':
                val = np.random.randint(0, self.n_vals)
                cards[:, i] = val
            else:
                vals = np.random.permutation(self.n_vals)
                cards[:, i] = vals
        return cards

    def _is_set(self, triplet: np.ndarray) -> bool:
        attr_sums = np.sum(triplet, axis=0)
        return np.all(attr_sums % self.n_vals == 0)