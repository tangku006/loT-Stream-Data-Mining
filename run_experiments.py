from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.classification.lazy.knn_adwin import KNNAdwin, KNN
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from skmultiflow.options.file_option import FileOption
from skmultiflow.data.file_stream import FileStream
from skmultiflow.classification.hoeffding_tree_multilabel import HoeffdingTreeMultiLabel

from sklearn.linear_model.perceptron import Perceptron
from skmultiflow.classification.multi_output_learner import MultiOutputLearner

dataset = "music"

# 1. Create a stream
opt = FileOption("FILE", "OPT_NAME", "./data/"+dataset+".csv", "CSV", False)
stream = FileStream(opt, 0, 6)
# 2. Prepare for use
stream.prepare_for_use()

# 3. Instantiate the HoeffdingTree classifier
h = [
        HoeffdingTreeMultiLabel(),
        MultiOutputLearner(h=Perceptron())
     ]
# 4. Setup the evaluator
eval = EvaluatePrequential(pretrain_size=50, output_file='result_'+dataset+'.csv', max_instances=10000, batch_size=1, n_wait=50, max_time=1000000000, task_type='multi_output', show_plot=False, plot_options=['hamming_score'])
# 5. Run
eval.eval(stream=stream, classifier=h)
