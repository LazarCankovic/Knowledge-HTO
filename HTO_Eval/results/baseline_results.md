Model	    F1	    Recall	Precision	Balanced Acc.	ROC-AUC	    AP
LR	       0.456	0.636	0.355	    0.592	        0.634	    0.404
LSTM	   0.462	0.742	0.336	    0.585	        0.627	    0.385


The LSTM achieved higher positive-class recall and slightly higher F1, but Logistic Regression produced stronger balanced accuracy, ROC-AUC, and average precision. This suggests that the current sequential representation does not yet provide a clear overall advantage over a simpler bag-of-events representation.

LSTM Experiment Explanation:
Experiment ID:
lstm_20261001T050025Z_a21efc5f

Git commit:
9dbf03026d750030cdbb2cc490abe048d82a8367

Split ID:
6441f1d0b131

Embedding:
128

Hidden:
128

Layers:
1

Max length:
512

Batch size:
64

LR:
0.001

Class weighting:
yes

Best validation epoch:
2

Final decision threshold:
0.45


