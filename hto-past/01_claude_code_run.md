

  Full end-to-end benchmark pipeline implemented and validated on a laptop. Every baseline trains, evaluates, and saves artifacts.                                                
                                                                                                                                                                                  
  What was built:                                                                                                                                                                 
                                                                                                                                                                                  
  - Canonical data builder (build_data.py → src/hto/data/build.py) — turns raw MIMIC parquets into per-patient records with visits carrying diag codes, med codes, lab numerics,  
  vital numerics, and day_offset (days since patient's first observed event).                                                                                                   
  - Four dataset variants (src/hto/data/variants.py): codes_visit, codes_flat, multimodal, text_sequence. Shared vocab fit on train only.                                         
  - Stable patient-level 85/15 split (split.json) shared across every variant — same patients in train vs. test everywhere.                                                       
  - Seven baselines in src/hto/models/, each with a uniform build_batch+forward interface:                                                                                        
    - Med-BERT (codes_flat) — BERT over flat code stream + segment/visit embedding                                                                                                
    - T-LSTM (codes_visit) — paper's exact cell with g(Δt)=1/log(e+Δt)                                                                                                            
    - Time2Vec (codes_flat) — learnable Fourier time feature fused with Transformer                                                                                               
    - TransformEHR (codes_visit) — hierarchical visit-then-patient encoder                                                                                                        
    - BEHRT (codes_visit) — [SEP]-joined flatten with visit/segment embeddings                                                                                                    
    - CEHR-BEHRT (codes_visit) — adds W/M/Y ATT tokens + continuous time                                                                                                          
    - Multimodal-BEHRT (multimodal) — adds numeric lab/vital branch with missingness masks                                                                                        
  - Shared training/eval (src/hto/train.py, src/hto/eval/metrics.py) — class-weighted CE, AUC/acc/F1/precision/recall/confusion, ROC plot, probability histogram, raw             
  logits+probs+preds in predictions.npz, per-epoch history in metrics.json.                                                                                                       
  - Registry + CLI (src/hto/registry.py, run.py) — python run.py --baseline X --dataset Y.                                                                                        
  - HPC submission: submit_HPC_jobs/run_one.sh (SLURM template, gpuqs/cs[001-004]/32G/4h, HF offline) + submit_all.sh (enumerates every baseline × dataset combo, sbatch-es each  
  with --export).                                                                                                                                                                 
  - Report aggregator (build_report.py) — walks predictions/ and emits reports/summary.md, summary.tsv, auc_bar.png, f1_bar.png.                                                  
  - Smoke suite (tests/smoke_all.sh) — builds a 200-patient sample and trains all 7 baselines in a few minutes on CPU.                                                            
  - Transformations doc (docs/transformations.md) — per-baseline: paper expectation, our adapter, processed file paths, caveats.                                                  
                                                                                                                                                                                  
  Smoke-test evidence (tests/smoke_all.sh):                                                                                                                                       
                                                                                                                                                                                  
  ┌────────────┬──────────────────┬─────────────┬───────┬───────┬───────┬────────┐                                                                                                
  │  dataset   │     baseline     │   variant   │  AUC  │  F1   │  acc  │ params │
  ├────────────┼──────────────────┼─────────────┼───────┼───────┼───────┼────────┤                                                                                                
  │ smoke_200  │ medbert          │ codes_flat  │ 0.671 │ 0.744 │ 0.633 │ 152k   │
  ├────────────┼──────────────────┼─────────────┼───────┼───────┼───────┼────────┤
  │ smoke_200  │ tlstm            │ codes_visit │ 0.478 │ 0.868 │ 0.767 │ 190k   │                                                                                                
  ├────────────┼──────────────────┼─────────────┼───────┼───────┼───────┼────────┤                                                                                                
  │ smoke_200  │ time2vec         │ codes_flat  │ 0.565 │ 0.816 │ 0.700 │ 150k   │                                                                                                
  ├────────────┼──────────────────┼─────────────┼───────┼───────┼───────┼────────┤                                                                                                
  │ smoke_200  │ transformehr     │ codes_visit │ 0.689 │ 0.885 │ 0.800 │ 263k   │
  ├────────────┼──────────────────┼─────────────┼───────┼───────┼───────┼────────┤                                                                                                
  │ smoke_200  │ behrt            │ codes_visit │ 0.453 │ 0.840 │ 0.733 │ 217k   │
  ├────────────┼──────────────────┼─────────────┼───────┼───────┼───────┼────────┤                                                                                                
  │ smoke_200  │ cehrbert         │ codes_visit │ 0.578 │ 0.863 │ 0.767 │ 217k   │
  ├────────────┼──────────────────┼─────────────┼───────┼───────┼───────┼────────┤                                                                                                
  │ smoke_200  │ multimodal_behrt │ multimodal  │ 0.609 │ 0.809 │ 0.700 │ 273k   │
  ├────────────┼──────────────────┼─────────────┼───────┼───────┼───────┼────────┤                                                                                                
  │ smoke_1000 │ transformehr     │ codes_visit │ 0.614 │ 0.691 │ 0.607 │ 584k   │
  ├────────────┼──────────────────┼─────────────┼───────┼───────┼───────┼────────┤                                                                                                
  │ smoke_1000 │ multimodal_behrt │ multimodal  │ 0.706 │ 0.749 │ 0.673 │ 602k   │
  └────────────┴──────────────────┴─────────────┴───────┴───────┴───────┴────────┘                                                                                                
                                                            
  (Numbers are expected-low on smoke; the smoke_1000 run shows train loss dropping monotonically and AUC rising across epochs — real learning signal.)                            
                                                            
  To run full scale on HPC:                                                                                                                                                       
                                                            
  # once, on a node with the parquets                                                                                                                                             
  python build_data.py --tag full
                                                                                                                                                                                  
  # fan out 7 jobs                                          
  bash submit_HPC_jobs/submit_all.sh full                                                                                                                                         
                                                                                                                                                                                  
  # aggregate after jobs complete
  python build_report.py                                                                                                                                                          



