from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.persistence.firestore import LocalJsonPacketRepository
from job_rejection_agent.services import DiagnosticService


ROOT = Path(__file__).resolve().parents[1]


class PersistenceTests(unittest.TestCase):
    def test_local_repository_round_trip(self) -> None:
        service = DiagnosticService()
        packet = service.diagnose(
            resume_path=ROOT / "fixtures" / "resumes" / "meera_data_analyst.txt",
            jd_text=(ROOT / "fixtures" / "jds" / "data_analyst_rotational.md").read_text(encoding="utf-8"),
            persist=False,
        ).packet
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = LocalJsonPacketRepository(storage_path=Path(temp_dir) / "packets.json")
            repo.save_packet(packet)
            loaded = repo.load_packet(packet.packet_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.packet_id, packet.packet_id)
            self.assertIsNotNone(loaded.report.rewritten_resume)

    def test_local_repository_round_trip_comparison(self) -> None:
        service = DiagnosticService()
        comparison = service.compare_job_descriptions(
            resume_path=ROOT / "fixtures" / "resumes" / "meera_data_analyst.txt",
            jd_texts=[
                (ROOT / "fixtures" / "jds" / "data_analyst_rotational.md").read_text(encoding="utf-8"),
                (ROOT / "fixtures" / "jds" / "ai_products_intern.md").read_text(encoding="utf-8"),
            ],
            user_id="repo-user",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = LocalJsonPacketRepository(storage_path=Path(temp_dir) / "packets.json")
            repo.save_comparison(comparison)
            loaded = repo.load_comparison(comparison.comparison_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.comparison_id, comparison.comparison_id)
            self.assertEqual(len(loaded.rows), 2)


if __name__ == "__main__":
    unittest.main()
