import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import sparkq


class TrainSessionsTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("tmux"), "tmux가 설치되어 있어야 합니다")
    def test_missing_tmux_server_is_an_empty_session_list(self):
        socket_name = f"sparkq-test-{uuid.uuid4().hex}"

        self.assertEqual(sparkq.train_sessions(tmux_socket=socket_name), [])


class KindLaneTest(unittest.TestCase):
    def test_side_limit_is_required_and_capped(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "queue.json").write_text(json.dumps({"kind": "queue", "run": "true"}))
            (directory / "missing.json").write_text(json.dumps({"kind": "missing", "lane": "side", "run": "true"}))
            (directory / "long.json").write_text(json.dumps({
                "kind": "long", "lane": "side", "limit_seconds": 3601, "run": "true",
            }))
            (directory / "side.json").write_text(json.dumps({
                "kind": "side", "lane": "side", "limit_seconds": 600, "run": "true",
            }))
            stderr = io.StringIO()
            with mock.patch.object(sparkq, "KINDS_DIR", directory), contextlib.redirect_stderr(stderr):
                kinds = sparkq.load_kinds()

        self.assertEqual(set(kinds), {"queue", "side"})
        self.assertEqual(kinds["queue"]["lane"], "queue")
        self.assertIn("missing.json", stderr.getvalue())
        self.assertIn("long.json", stderr.getvalue())

    def test_side_session_can_never_be_train_prefixed(self):
        spec = {"lane": "side", "session": "train-viewer-${id}"}

        self.assertEqual(sparkq._session_name(spec, {"id": "123"}), "side-viewer-123")


class StepSecondsTest(unittest.TestCase):
    def test_tqdm_rates_are_normalized_to_seconds_per_step(self):
        fast = sparkq.parse_progress("lerobot", ["1/20 [00:01<00:19, 1.14step/s]"])
        slow = sparkq.parse_progress("lerobot", ["1/20 [00:03<01:05, 3.45s/step]"])

        self.assertAlmostEqual(fast["step_seconds"], 1 / 1.14)
        self.assertEqual(slow["step_seconds"], 3.45)


class ExitStatusTest(unittest.TestCase):
    def test_run_script_preserves_failures_and_rejects_zero_with_traceback(self):
        cases = (
            ("true", 0),
            ("exit 7", 7),
            ("printf 'Traceback (most recent call last):\\nRuntimeError: probe\\n'", 1),
        )
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            for index, (command, expected) in enumerate(cases):
                with self.subTest(command=command):
                    log = directory / f"{index}.log"
                    script = directory / f"{index}.sh"
                    script.write_text(sparkq.run_script_text(command, log))
                    with log.open("wb") as output:
                        result = subprocess.run(["bash", script], stdout=output, stderr=subprocess.STDOUT)
                    self.assertEqual(result.returncode, expected)


class SideIsolationTest(unittest.TestCase):
    def test_side_processes_are_removed_from_gpu_apps(self):
        apps = [{"pid": "10"}, {"pid": "20"}]
        with mock.patch.object(sparkq, "session_process_ids", return_value={"10"}):
            self.assertEqual(sparkq.without_session_apps(apps, "side-viewer"), [{"pid": "20"}])

    def test_second_side_is_a_conflict(self):
        with mock.patch.object(sparkq, "current_side", return_value={"id": "already-running"}):
            with self.assertRaises(sparkq.Conflict):
                sparkq._start_side("isaac-play", {})

    def test_queue_gate_ignores_the_side_session_gpu_process(self):
        queued_file = mock.Mock()
        queued_job = {"id": "queued", "session": "train-queued"}
        side = {"id": "viewer", "session": "side-viewer", "expires_at": 10**20}
        patches = (
            mock.patch.object(sparkq, "current_side", return_value=side),
            mock.patch.object(sparkq, "current_job", return_value=None),
            mock.patch.object(sparkq, "session_alive", return_value=True),
            mock.patch.object(sparkq, "queued_files", return_value=[queued_file]),
            mock.patch.object(sparkq, "train_sessions", return_value=[]),
            mock.patch.object(sparkq, "compute_apps", return_value=[{"pid": "10"}]),
            mock.patch.object(sparkq, "session_process_ids", return_value={"10"}),
            mock.patch.object(sparkq, "read_json", return_value=queued_job),
            mock.patch.object(sparkq, "write_json"),
            mock.patch.object(sparkq, "start", return_value=queued_job),
            mock.patch.object(sparkq, "PAUSED_FILE", Path(f"/tmp/sparkq-no-pause-{uuid.uuid4().hex}")),
        )
        with contextlib.ExitStack() as stack:
            entered = [stack.enter_context(patch) for patch in patches]
            sparkq._tick()

        entered[-2].assert_called_once_with(queued_job)

    def test_side_deadlines_are_persisted_and_extend_at_most_one_hour(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            kinds_dir = root / "kinds"
            kinds_dir.mkdir()
            (kinds_dir / "viewer.json").write_text(json.dumps({
                "kind": "viewer", "lane": "side", "limit_seconds": 600,
                "title": "Viewer", "session": "train-viewer-${id}", "fields": [], "run": "true",
            }))
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(sparkq, "ROOT", root))
                stack.enter_context(mock.patch.object(sparkq, "RUNS_DIR", root / "runs"))
                stack.enter_context(mock.patch.object(sparkq, "CURRENT_SIDE_FILE", root / "current_side"))
                stack.enter_context(mock.patch.object(sparkq, "KINDS_DIR", kinds_dir))
                stack.enter_context(mock.patch.object(sparkq, "current_job", return_value=None))
                run = stack.enter_context(mock.patch.object(sparkq.subprocess, "run"))

                side = sparkq.start_side("viewer", {})
                extended = sparkq.extend_side(600)
                with self.assertRaises(sparkq.Invalid):
                    sparkq.extend_side(3001)

            self.assertTrue(side["session"].startswith("side-"))
            self.assertAlmostEqual(side["expires_at"] - side["started_at"], 600)
            self.assertAlmostEqual(side["extendable_until"] - side["started_at"], 3600)
            self.assertAlmostEqual(extended["expires_at"] - side["expires_at"], 600)
            job_id = (root / "current_side").read_text().strip()
            wrapper = (root / "runs" / job_id / "run.sh").read_text()
            self.assertIn("timeout --signal=INT 3600", wrapper)
            run.assert_called_once()

    def test_http_maps_side_conflict_and_limit_errors(self):
        for error, status in ((sparkq.Conflict("busy"), 409), (sparkq.Invalid("limit"), 400)):
            with self.subTest(status=status):
                handler = object.__new__(sparkq.Handler)
                handler._route = mock.Mock(side_effect=error)
                handler._send = mock.Mock()

                handler._dispatch()

                self.assertEqual(handler._send.call_args.args[1], status)


if __name__ == "__main__":
    unittest.main()
