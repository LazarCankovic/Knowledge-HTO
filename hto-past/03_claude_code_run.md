  What got built this round:                                                                                                                                                                                                                     
                                                 
  - multidomain_visit variant (src/hto/data/variants.py) — mirrors the final_sequences.parquet grammar ([EOV] [EOD] [EOM] [EOL] [EOVI] delimiters) but re-attaches day offsets and visit gaps to every event; emits per-token domain IDs and a   
  parallel numeric stream.                                                                                                                                                                                                                       
  - Composite PE (src/hto/models/composite_pe.py) — sinusoidal(pos) + T_abs(day) + T_gap(Δday) + D(domain) + DT(domain,day), each component toggleable via --pe-parts.
  - Relative-day attention bias + BiasableEncoder — per-head φ(|Δd_i − Δd_j|) threaded into every layer, toggled via --rel-day-bias.                                                                                                             
  - 7 multidomain baselines (medbert_md, tlstm_md, time2vec_md, transformehr_md, behrt_md, cehrbert_md, multimodal_behrt_md) — all use multidomain_visit + composite PE + optional rel-day bias.                                                 
  - 2 new irregular-TS baselines: mtan (Shukla & Marlin ICLR'21), strats (Tipirneni & Reddy KDD'22).                                                                                                                                             
  - Metrics extended: AUPRC, Brier, ECE, per-target-med breakdown, PR plot.                                                                                                                                                                      
  - HPC: submit_all.sh now covers all 16 baselines and supports PE_ABLATION=1 sweep.                                                                                                                                                             
  - Smoke suite: all 16 baselines pass on smoke_1000 (850 train / 150 test, 2 epochs, CPU).                                                                                                                                                      
                                                                                                                                                                                                                                                 
  Smoke_1000 headline results (2 epochs on CPU, d_model=64, full multidomain input):                                                                                                                                                             
                                                                                                                                                                                                                                                 
  ┌──────┬──────────────────┬───────┬───────┬───────┐                                                                                                                                                                                            
  │ rank │     baseline     │  AUC  │ AUPRC │ Brier │                                                                                                                                                                                            
  ├──────┼──────────────────┼───────┼───────┼───────┤                                                                                                                                                                                            
  │ 1    │ strats           │ 0.791 │ 0.927 │ 0.189 │                                                                                                                                                                                            
  ├──────┼──────────────────┼───────┼───────┼───────┤       
  │ 2    │ mtan             │ 0.703 │ 0.887 │ 0.221 │                                                                                                                                                                                            
  ├──────┼──────────────────┼───────┼───────┼───────┤       
  │ 3    │ time2vec         │ 0.694 │ 0.867 │ 0.222 │
  ├──────┼──────────────────┼───────┼───────┼───────┤                                                                                                                                                                                            
  │ 4    │ multimodal_behrt │ 0.680 │ 0.863 │ 0.220 │
  ├──────┼──────────────────┼───────┼───────┼───────┤                                                                                                                                                                                            
  │ 5    │ time2vec_md      │ 0.654 │ 0.820 │ 0.210 │                                                                                                                                                                                            
  ├──────┼──────────────────┼───────┼───────┼───────┤
  │ 6    │ medbert          │ 0.601 │ 0.836 │ 0.232 │                                                                                                                                                                                            
  └──────┴──────────────────┴───────┴───────┴───────┘       
                                                                                                                                                                                                                                                 
  STraTS's triplet-set-attention is the standout — exactly what we'd expect from an irregular-multivariate-TS model on this data. The _md family is close to their originals but hasn't pulled away on 2-epoch CPU smoke; real gains should show
  with full training on HPC.                                                                                                                                                                                                                     
   
  Ready for HPC: python build_data.py --tag full && bash submit_HPC_jobs/submit_all.sh full fans out 16 jobs; add PE_ABLATION=1 for the composite-PE ablation sweep.                                                                             
