Field	        Type	                    Description
patient_id	integer	MIMIC       patient identifier; used only for patient-level splitting/tracking
sequence	list[str]	        Chronologically ordered diagnosis and medication tokens through the LAST observed HTO cutoff
label	    integer	            Binary outcome: 1 if 90-day median SBP <120 and DBP <80; otherwise 0

Token Formats(DX.. = Diagnosis, MED... = What HT medication was used):
DX_10_I10 
DX_9_4019
MED_AMLODIPINE
MED_LISINOPRIL

For LSTM (patient_id is NEVER used as a predictive feature):
[PAD] = model padding
[UNK] = unseen token




