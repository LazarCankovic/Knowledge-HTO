

- in baselines/baseline_list.md
we have a list of baselines that we want to implement (their code, paper, name) 


- in data_preprocessing/MIMIC.ipynb there is a preprocessing script that we used to process original data to 

mimic_data outputs: 

final_medications.parquet

all_mapped_medications.parquet

final_diagnoses.parquet

final_vitals.parquet

final_labs.parquet

final_sequences.parquet

final_patients_table.parquet


---------------

based on the existing data we have in mimic_data outputs, 
implement all the baselines from the baseline_list.md -- adjust the input format that is needed for these baselines based on their input data description in the respective papers and code repositories. Make sure the time difference  is handled properly (if applicable) because in our dataset - (mimic_data outputs) - time is measured as days since first observed visit vs in some of these baselines they have different formats... 




---------------


Implement the following code:
1. Data processing for each of the baselines
2. Run model for each of the baselines on that particular processed data 
3. Evaluation of the model in terms of (AUC, accuracy, F1, prec, recall, confusion matrix) and respective nicest looking visualizations 
4. Make sure the run script is generalizable to pick on parameter for the model vs dataset variation (we may want to run our models on these different dataset variations - so make sure to save these different dataset sequence variants separately such that we can do baseline x dataset runs if needed) and wrap this up in a nice evaluation pipeline as well... 
5. Make sure that you split test-train (15-85%) BUT make sure you keep the same patients in test vs train across these different variants 
6.  Finally, this is going to be run on an HPC machine - so we want to run these experiments in parallel on different cluster nodes (so include another submit script that is going to specify what all different baseline x dataset combinations we want to run and submit jobs for each of these combinations in different GPU queues). Take a look at submit_HPC_jobs/run_job_hpc.sh and use this as a guide/template for submitting jobs for each of those baselines.
7. Please keep a document on what all changes you had to perform on our datasets for each of the baselines - where those datasets are stored and what the original model for those datasets are; save the raw predictions (logits, probability scores, as well as predicted labels) for each of the models x datasets --- and save the results in a nicely formatted report... 






