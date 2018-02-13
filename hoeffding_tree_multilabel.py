__author__ = 'Jacob Montiel'

import sys
import logging
import textwrap
from abc import ABCMeta
from operator import attrgetter
from skmultiflow.classification.core.attribute_class_observers.gaussian_numeric_attribute_class_observer \
    import GaussianNumericAttributeClassObserver
from skmultiflow.classification.core.attribute_class_observers.nominal_attribute_class_observer \
    import NominalAttributeClassObserver
from skmultiflow.classification.core.attribute_class_observers.null_attribute_class_observer \
    import NullAttributeClassObserver
from skmultiflow.classification.core.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.classification.core.split_criteria.gini_split_criterion import GiniSplitCriterion
from skmultiflow.classification.core.split_criteria.info_gain_ml_split_criterion import InfoGainSplitMLCriterion
from skmultiflow.classification.core.utils.utils import do_naive_bayes_prediction
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree

GINI_SPLIT = 'gini'
INFO_GAIN_SPLIT_ML = 'info_gain_ml'
MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'

# logger
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class HoeffdingTreeMultiLabel(HoeffdingTree):
    class FoundNode(object):
        def __init__(self, node=None, parent=None, parent_branch=None):
            self.node = node
            self.parent = parent
            self.parent_branch = parent_branch

    class Node(metaclass=ABCMeta):
        def __init__(self, class_observations=None):
            if class_observations is None:
                class_observations = {}  # Dictionary (class_value, weight)
            self._observed_class_distribution = class_observations

        @staticmethod
        def is_leaf():
            return True

        def filter_instance_to_leaf(self, X, parent, parent_branch):
            return HoeffdingTreeMultiLabel.FoundNode(self, parent, parent_branch)

        def get_observed_class_distribution(self):
            return self._observed_class_distribution

        def get_class_votes(self, X, ht):
            return self._observed_class_distribution

        def observed_class_distribution_is_pure(self):
            count = 0
            for _, weight in self._observed_class_distribution.items():
                if weight is not 0:
                    count += 1
                    if count == 2:  # No need to count beyond this point
                        break
            return count < 2

        def subtree_depth(self):
            return 0

        def calculate_promise(self):
            total_seen = sum(self._observed_class_distribution.values())
            if total_seen > 0:
                return total_seen - max(self._observed_class_distribution.values())
            else:
                return 0

        def __sizeof__(self):
            return object.__sizeof__(self) + sys.getsizeof(self._observed_class_distribution)

        def calc_byte_size_including_subtree(self):
            return self.__sizeof__()

        def describe_subtree(self, ht, buffer, indent=0):
            buffer[0] += textwrap.indent('Leaf = ', ' ' * indent)
            class_val = max(self._observed_class_distribution, key=self._observed_class_distribution.get)
            buffer[0] += 'Class {} | {}\n'.format(class_val, self._observed_class_distribution)

        def get_description(self):
            pass

    class SplitNode(Node):
        def __init__(self, split_test, class_observations, size=-1):
            super().__init__(class_observations)
            self._split_test = split_test
            # Dict of tuples (branch, child)
            if size > 0:
                self._children = [None] * size
            else:
                self._children = []

        def num_children(self):
            return len(self._children)

        def set_child(self, index, node):
            if self._split_test.max_branches() >= 0 and index >= self._split_test.max_branches():
                raise IndexError
            self._children[index] = node

        def get_child(self, index):
            return self._children[index]

        def instance_child_index(self, X):
            return self._split_test.branch_for_instance(X)

        @staticmethod
        def is_leaf():
            return False

        def filter_instance_to_leaf(self, X, parent, parent_branch):
            child_index = self.instance_child_index(X)
            if child_index >= 0:
                child = self.get_child(child_index)
                if child is not None:
                    return child.filter_instance_to_leaf(X, self, child_index)
                else:
                    return HoeffdingTreeMultiLabel.FoundNode(None, self, child_index)
            else:
                return HoeffdingTreeMultiLabel.FoundNode(self, parent, parent_branch)

        def subtree_depth(self):
            max_child_depth = 0
            for child in self._children:
                if child is not None:
                    depth = child.subtree_depth()
                    if depth > max_child_depth:
                        max_child_depth = depth
            return max_child_depth + 1

        def __sizeof__(self):
            return object.__sizeof__(self) + sys.getsizeof(self._children) + sys.getsizeof(self._split_test)

        def calc_byte_size_including_subtree(self):
            byte_size = self.__sizeof__()
            for child in self._children:
                if child is not None:
                    byte_size += child.calc_byte_size_including_subtree()
            return byte_size

        def describe_subtree(self, ht, buffer, indent=0):
            for branch_idx in range(self.num_children()):
                child = self.get_child(branch_idx)
                if child is not None:
                    buffer[0] += textwrap.indent('if ', ' ' * indent)
                    buffer[0] += self._split_test.describe_condition_for_branch(branch_idx)
                    buffer[0] += ':\n'
                    child.describe_subtree(ht, buffer, indent + 2)

    class LearningNode(Node):
        def __init__(self, initial_class_observations=None):
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, ht):
            pass

    class InactiveLearningNode(LearningNode):
        def __init__(self, initial_class_observations=None):
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, ht):
            if tuple(y) not in self._observed_class_distribution:
                self._observed_class_distribution[tuple(y)] = 0.0
            self._observed_class_distribution[tuple(y)] += weight

    class ActiveLearningNode(LearningNode):
        """A Hoeffding Tree node that supports growth."""
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)
            self._weight_seen_at_last_split_evaluation = self.get_weight_seen()
            self._is_initialized = False
            self._attribute_observers = []

        def learn_from_instance(self, X, y, weight, ht):
            if not self._is_initialized:
                self._attribute_observers = [None] * len(X)
                self._is_initialized = True
            if tuple(y) not in self._observed_class_distribution:
                self._observed_class_distribution[tuple(y)] = 0.0
            self._observed_class_distribution[tuple(y)] += weight

            for i in range(len(X)):
                obs = self._attribute_observers[i]
                if obs is None:
                    if i in ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], tuple(y), weight)

        def get_weight_seen(self):
            return sum(self._observed_class_distribution.values())

        def get_weight_seen_at_last_split_evaluation(self):
            return self._weight_seen_at_last_split_evaluation

        def set_weight_seen_at_last_split_evaluation(self, weight):
            self._weight_seen_at_last_split_evaluation = weight

        def get_best_split_suggestions(self, criterion, ht):
            best_suggestions = []
            pre_split_dist = self._observed_class_distribution
            if not ht.no_preprune:
                # Add null split as an option
                null_split = AttributeSplitSuggestion(None, [{}],
                                                      criterion.get_merit_of_split(pre_split_dist, [pre_split_dist]))
                best_suggestions.append(null_split)
            for i, obs in enumerate(self._attribute_observers):
                best_suggestion = obs.get_best_evaluated_split_suggestion(criterion, pre_split_dist,
                                                                          i, ht.binary_split)
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)
            return best_suggestions

        def disable_attribute(self, att_idx):
            if att_idx < len(self._attribute_observers) and att_idx > 0:
                self._attribute_observers[att_idx] = NullAttributeClassObserver()

    class LearningNodeNB(ActiveLearningNode):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)

        def get_class_votes(self, X, ht):
            if self.get_weight_seen() >= ht.nb_threshold:
                return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            else:
                return super().get_class_votes(X, ht)

        def disable_attribute(self, att_index):
            pass

    class LearningNodeNBAdaptive(LearningNodeNB):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, ht):
            if self._observed_class_distribution != {} and max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight, ht)

        def get_class_votes(self, X, ht):
            if self._mc_correct_weight > self._nb_correct_weight:
                return self._observed_class_distribution
            return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

    # ================================================
    # == Multi-label Hoeffding Tree implementation ===
    # ================================================
    def __init__(self, max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
                 split_criterion='info_gain_ml', split_confidence=0.0000001, tie_threshold=0.05, binary_split=False,
                 stop_mem_management=False, remove_poor_atts=False, no_preprune=False, leaf_prediction='mc',
                 nb_threshold=0, nominal_attributes=None):
        super().__init__()
        self.max_byte_size = max_byte_size
        self.memory_estimate_period = memory_estimate_period
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.stop_mem_management = stop_mem_management
        self.remove_poor_atts = remove_poor_atts
        self.no_preprune = no_preprune
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes
        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        self._train_weight_seen_by_model = 0.0

    @property
    def split_criterion(self):
        return self._split_criterion

    @split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion != GINI_SPLIT and split_criterion != INFO_GAIN_SPLIT_ML:
            logger.info("Invalid option {}', will use default '{}'".format(split_criterion, INFO_GAIN_SPLIT_ML))
            self._split_criterion = INFO_GAIN_SPLIT_ML
        else:
            self._split_criterion = split_criterion

    def _attempt_to_split(self, node: ActiveLearningNode, parent: SplitNode, parent_idx: int):
        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == INFO_GAIN_SPLIT_ML:
                split_criterion = InfoGainSplitMLCriterion()
            else:
                split_criterion = InfoGainSplitMLCriterion()
            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort(key=attrgetter('merit'))
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                    node.get_observed_class_distribution()), self.split_confidence, node.get_weight_seen())
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                        or hoeffding_bound < self.tie_threshold):    # best_suggestion.merit > 1e-10 and \
                    should_split = True
                if self.remove_poor_atts is not None and self.remove_poor_atts:
                    poor_atts = set()
                    # Scan 1 - add any poor attribute to set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            split_atts = best_split_suggestions[i].split_test.get_atts_test_depends_on()
                            if len(split_atts) == 1:
                                if best_suggestion.merit - best_split_suggestions[i].merit > hoeffding_bound:
                                    poor_atts.add(int(split_atts[0]))
                    # Scan 2 - remove good attributes from set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            split_atts = best_split_suggestions[i].split_test.get_atts_test_depends_on()
                            if len(split_atts) == 1:
                                if best_suggestion.merit - best_split_suggestions[i].merit < hoeffding_bound:
                                    poor_atts.remove(int(split_atts[0]))
                    for poor_att in poor_atts:
                        node.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.split_test is None:
                    # Preprune - null wins
                    self._deactivate_learning_node(node, parent, parent_idx)
                else:
                    new_split = self.new_split_node(split_decision.split_test,
                                                    node.get_observed_class_distribution(),
                                                    split_decision.num_splits())
                    for i in range(split_decision.num_splits()):
                        new_child = self._new_learning_node(split_decision.resulting_class_distribution_from_split(i))
                        new_split.set_child(i, new_child)
                    self._active_leaf_node_cnt -= 1
                    self._decision_node_cnt += 1
                    self._active_leaf_node_cnt += split_decision.num_splits()
                    if parent is None:
                        self._tree_root = new_split
                    else:
                        parent.set_child(parent_idx, new_split)
                # Manage memory
                self.enforce_tracker_limit()
