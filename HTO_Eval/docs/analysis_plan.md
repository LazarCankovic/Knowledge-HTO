Baseline analysis:
Majority-class baseline
        ↓
TF-IDF Logistic Regression
        ↓
LSTM


This tells us whether event ordering(sequence) gives us better predictions than just simple event preesence.


Next Steps(Improvemets):
1. Add explicit visit-boundary tokens
2. Investigate vocabulary frequency filtering
3. Evaluate removal of class weighting
4. Add pre-index BP history as an input modality
5. Evaluate stronger sequence models:
   - BiLSTM + attention
   - BEHRT / Transformer
6. Later integrate the hypertension knowledge base
