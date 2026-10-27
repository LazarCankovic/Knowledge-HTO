"""Small RF2 fixtures verify the lab → LOINC → SNOMED → diagnosis join."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.icd_snomed_map import INFERRED  # noqa: E402
from hto.kg.lab_snomed_graph import build_lab_snomed_graph  # noqa: E402
from hto.kg.patient_graphs import build_all  # noqa: E402
from hto.kg.snomed_graph import IS_A, save_graph, load_graph  # noqa: E402
import build_lab_kg  # noqa: E402


def rf2(root: Path, prefix: str, rows: list[dict]) -> None:
    directory = root / "Snapshot" / "Terminology"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / f"{prefix}_Snapshot_TEST.txt", sep="\t", index=False)


def fixture(root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, Path, Path]:
    base = {
        "entities": pd.DataFrame([(1, "10_I10", "icd", "Hypertension"),
                                  (2, "100", "snomed", "Finding"),
                                  (3, "200", "snomed", "Shared concept")],
                                 columns=["id", "key", "kind", "name"]),
        "relations": pd.DataFrame([(1, "maps_to"), (2, "inv_maps_to"),
                                   (3, "is_a"), (4, "inv_is_a")],
                                  columns=["id", "name"]),
        "triples": pd.DataFrame([(1, 1, 2), (2, 2, 1), (2, 3, 3), (3, 4, 2)],
                                columns=["h", "r", "t"]),
        "stats": {},
    }
    core, ext = root / "core", root / "extension"
    rf2(core, "sct2_Relationship", [dict(active="1", sourceId="100", destinationId="200",
                                          typeId=IS_A, characteristicTypeId=INFERRED)])
    rf2(ext, "sct2_Relationship", [dict(active="1", sourceId="300", destinationId="200",
                                         typeId=IS_A, characteristicTypeId=INFERRED)])
    rf2(ext, "sct2_Identifier", [dict(alternateIdentifier="1234-5", active="1",
                                       identifierSchemeId="30051010000102", referencedComponentId="300"),
                                 dict(alternateIdentifier="9999-9", active="0",
                                      identifierSchemeId="30051010000102", referencedComponentId="301")])
    rf2(ext, "sct2_Concept", [dict(id="300", active="1"), dict(id="301", active="1")])
    for release in (core, ext):
        rf2(release, "sct2_Description", [dict(active="1", conceptId="300", typeId="900000000000003001",
                                                term="Lab observation")])
    items = pd.DataFrame([dict(itemid=11, label="Glucose", fluid="Blood", category="Chemistry"),
                          dict(itemid=12, label="Sodium", fluid="Blood", category="Chemistry")])
    mapping = pd.DataFrame([dict(itemid=11, loinc_code="1234-5"),
                            dict(itemid=12, loinc_code="9999-9")])
    return base, items, mapping, core, ext


class LabSnomedGraphTest(unittest.TestCase):
    def test_shared_snomed_id_and_patient_cross_domain_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base, items, mapping, core, ext = fixture(root)
            graph = build_lab_snomed_graph(base, items, mapping, core, ext, ancestor_depth=2)
            self.assertEqual(graph["stats"]["lab_codes_with_active_snomed"], 1)
            self.assertEqual(graph["entities"].iloc[:3]["key"].tolist(), ["10_I10", "100", "200"])
            ids = dict(zip(graph["entities"]["key"], graph["entities"]["id"]))
            rels = dict(zip(graph["relations"]["name"], graph["relations"]["id"]))
            triples = set(map(tuple, graph["triples"][["h", "r", "t"]].to_numpy()))
            self.assertIn((ids["lab:11"], rels["maps_lab_to_snomed"], ids["300"]), triples)
            self.assertIn((ids["lab:11"], rels["maps_to_loinc"], ids["loinc:1234-5"]), triples)
            self.assertIn((ids["loinc:1234-5"], rels["loinc_snomed_identifier"], ids["300"]), triples)
            self.assertIn((ids["300"], rels["is_a"], ids["200"]), triples)
            self.assertIn((1, 1, 2), triples)  # the original diagnosis link survives
            self.assertNotIn((ids["lab:12"], rels["maps_lab_to_snomed"], ids["300"]), triples)
            save_graph(graph, root / "out")
            loaded = load_graph(root / "out")
            events = pd.DataFrame([dict(patient_id=1, time="2020-01-01", code="10_I10"),
                                   dict(patient_id=1, time="2020-01-04", code="lab:11")])
            patients, stats = build_all(events, loaded, code_col="code", code_kind="mixed",
                                        path_time_priority=True)
            self.assertEqual(stats["patients_with_lab_diagnosis_path"], 1)
            self.assertIn("10_I10", patients[1]["code_index"])
            self.assertIn("lab:11", patients[1]["code_index"])
            self.assertIn(3, patients[1]["edge_tau"])

    def test_missing_reviewed_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base, items, mapping, core, ext = fixture(Path(d))
            with self.assertRaisesRegex(ValueError, "No lab items map"):
                build_lab_snomed_graph(base, items, mapping.assign(loinc_code="unknown"), core, ext)

    def test_cli_uses_official_mimic_code_loinc_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base, items, mapping, core, ext = fixture(root)
            save_graph(base, root / "processed_data" / "kg" / "diag")
            items_path = root / "d_labitems.csv"
            items.to_csv(items_path, index=False)
            mapping_path = root / "d_labitems_to_loinc.csv"
            pd.DataFrame({"itemid (omop_source_code)": [11, 12, 13],
                          "omop_vocabulary_id": ["LOINC", "LOINC", "SNOMED"],
                          "omop_concept_code": ["1234-5", "9999-9", "300"]}).to_csv(mapping_path, index=False)
            args = ["build_lab_kg.py", "graph", "--tag", "lab", "--diagnosis-kg-tag", "diag",
                    "--labitems", str(items_path), "--loinc-map", str(mapping_path),
                    "--snomed-dir", str(core),
                    "--loinc-extension-dir", str(ext)]
            with patch.object(build_lab_kg, "ROOT", root), patch.object(sys, "argv", args):
                build_lab_kg.main()
            graph = load_graph(root / "processed_data" / "kg" / "lab")
            self.assertIn("maps_lab_to_snomed", set(graph["relations"]["name"]))


if __name__ == "__main__":
    unittest.main()
