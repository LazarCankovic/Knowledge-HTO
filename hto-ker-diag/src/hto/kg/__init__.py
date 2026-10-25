"""SNOMED CT knowledge graph for the KG-enriched diagnosis tower (htr_former_kg).

icd_snomed_map  ICD-9-CM / ICD-10-CM codes -> SNOMED CT concepts
snomed_graph    global graph: ICD nodes + mapped concepts + their SNOMED neighbourhood
transr          TransR pretraining of entity / relation embeddings on the global graph

Built from public files only (SNOMED CT US edition, CMS GEMs, an ICD code list), so it can be built
without MIMIC patient data. The outputs are SNOMED-derived: keep them under processed_data/ (gitignored).
"""
