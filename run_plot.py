import pandas as pd
from matplotlib.pyplot import *

dataset = "music"

df = pd.read_csv('result_'+dataset+'.csv',comment='#')
#ax = df.plot(x="x_count", y=["global_performance_0","global_performance_1","global_performance_2"], rot=45, linewidth=3, title=dataset)
ax = df.plot(x="x_count", y=["global_hamming_score_0","global_hamming_score_1"], rot=30, linewidth=2, title=dataset)
ax.set_xlabel("")
ax.set_title("Performance on the %s dataset" % dataset)
ax.legend([r"Multilabel Hoeffding Tree",r"Multi Output learner"], loc='best')
print("write out to %s ..." % dataset+".pdf")
savefig("result_"+dataset+".pdf")
show()
