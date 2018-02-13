__author__ = 'Jacob Montiel'

from skmultiflow.classification.core.split_criteria.info_gain_split_criterion import InfoGainSplitCriterion

import numpy as np

class InfoGainSplitMLCriterion(InfoGainSplitCriterion):
    """ InfoGainMLSplitCriterion

        Information Gain for multi-label classification
    """
    def __init__(self, min_branch_frac_option=0.01):
        super().__init__(min_branch_frac_option)

    @staticmethod
    def _compute_entropy_dict(dist):
        dist_r = {}
        for key,value in dist.items():
            dist_r[key] = 1-value
        return InfoGainSplitCriterion._compute_entropy_dict(dist)-InfoGainSplitCriterion._compute_entropy_dict(dist_r)
