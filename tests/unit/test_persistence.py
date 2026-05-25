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


if __name__ == "__main__":
    unittest.main()
