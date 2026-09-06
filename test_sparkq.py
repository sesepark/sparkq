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
        fast = sparkq.parse_progress("lerobot", ["Training: 5%| | 1/20 [00:01<00:19, 1.14step/s]"])
        slow = sparkq.parse_progress("lerobot", ["Training: 5%| | 1/20 [00:03<01:05, 3.45s/step]"])

        self.assertAlmostEqual(fast["step_seconds"], 1 / 1.14)
        self.assertEqual(slow["step_seconds"], 3.45)


class LerobotProgressTest(unittest.TestCase):
    def test_only_the_training_bar_counts_as_progress(self):
        """lerobot은 학습 말고도 tqdm을 쓴다.

        SmolVLA는 시작하면서 `Loading weights: 489/489`을 찍는다. 그것을 진행으로 읽으면
        화면이 시작하자마자 100%가 되는데, 그것은 사람이 보고 끝난 줄 알 자리다.
        """
        loading = "Loading weights: 100%|██████████| 489/489 [00:00<00:00, 5417.80it/s]"

        self.assertEqual(sparkq.parse_progress("lerobot", [loading]), {})

        started = sparkq.parse_progress("lerobot", [
            loading, "Training:   1%|          | 9/1000 [00:54<1:19:56,  4.84s/step]",
        ])
        self.assertEqual((started["step"], started["steps"]), (9, 1000))

    def test_a_resumed_run_counts_this_segment_not_the_running_total(self):
        """이어붙인 학습에서 `step:` 줄은 통산이고 tqdm은 이번 구간이다.

        둘 중 큰 쪽을 고르면 5,000짜리 막대에 통산 5,100이 들어가 진행률이 100%를 넘는다.
        `이어서 한 밤 더`를 건 사람이 알고 싶은 것은 오늘 밤이 어디까지 갔는가다.
        """
        found = sparkq.parse_progress("lerobot", [
            "step:5K loss:0.123 updt_s:4.5 data_s:0.2",
            "Training:   2%|▏         | 100/5000 [08:00<6:32:00,  4.80s/step]",
        ])

        self.assertEqual((found["step"], found["steps"]), (100, 5000))
        self.assertEqual(found["loss"], 0.123)

    def test_without_a_bar_the_step_line_still_reports_a_step_but_no_total(self):
        """총량을 모르면 지어내지 않는다. 0%에 붙은 막대는 `진행이 없다`로 읽힌다."""
        found = sparkq.parse_progress("lerobot", ["step:2K loss:0.5 updt_s:4.5 data_s:0.2"])

        self.assertEqual(found["step"], 2000)
        self.assertNotIn("steps", found)


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
    def test_recent_includes_failed_side_but_not_successful_side(self):
        with tempfile.TemporaryDirectory() as raw:
            runs_dir = Path(raw)
            jobs = (
                {"id": "queue-done", "lane": "queue", "state": "done", "finished_at": 1},
                {"id": "side-done", "lane": "side", "state": "done", "finished_at": 2},
                {"id": "side-failed", "lane": "side", "state": "failed", "finished_at": 3},
            )
            for job in jobs:
                directory = runs_dir / job["id"]
                directory.mkdir()
                (directory / "job.json").write_text(json.dumps(job))

            with mock.patch.object(sparkq, "RUNS_DIR", runs_dir):
                recent = sparkq.recent()

        self.assertEqual([job["id"] for job in recent], ["side-failed", "queue-done"])

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
            # 이 시험이 보는 것은 GPU를 볼 수 있는 기계의 문지기다. 맥에서 돌려도 그
            # 경로를 지나도록 깃발을 켜 둔다 — 안 그러면 다른 이유로 통과한다.
            mock.patch.object(sparkq, "WATCHES_GPU_PROCESSES", True),
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
                stack.enter_context(mock.patch.object(
                    sparkq.probe, "timeout_prefix", return_value="timeout --signal=INT 3600",
                ))

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

    def test_side_is_refused_when_the_deadline_cannot_be_enforced(self):
        """시한을 못 걸면 곁다리를 아예 띄우지 않는다.

        곁다리를 만든 이유가 시한 하나다 — 사고의 원인은 누가 껐느냐가 아니라 아무도 안
        껐다는 것이었다. 시한 없이 뜨는 곁다리는 막으려던 그 사고 자체다.
        """
        with mock.patch.object(sparkq, "current_side", return_value=None), \
                mock.patch.object(sparkq.probe, "timeout_prefix", return_value=None):
            with self.assertRaises(sparkq.Invalid) as caught:
                sparkq._start_side("isaac-play", {})

        self.assertIn("timeout", str(caught.exception))


class PlatformGateTest(unittest.TestCase):
    """GPU를 볼 수 없는 기계에서 문지기와 API가 어떻게 달라지는가."""

    def _tick_with(self, watches: bool, compute_apps):
        started = mock.Mock(return_value={"id": "queued"})
        patches = (
            mock.patch.object(sparkq, "WATCHES_GPU_PROCESSES", watches),
            mock.patch.object(sparkq, "current_side", return_value=None),
            mock.patch.object(sparkq, "current_job", return_value=None),
            mock.patch.object(sparkq, "queued_files", return_value=[mock.Mock()]),
            mock.patch.object(sparkq, "train_sessions", return_value=[]),
            mock.patch.object(sparkq, "compute_apps", compute_apps),
            mock.patch.object(sparkq, "read_json", return_value={"id": "queued"}),
            mock.patch.object(sparkq, "write_json"),
            mock.patch.object(sparkq, "start", started),
            mock.patch.object(sparkq, "PAUSED_FILE", Path(f"/tmp/sparkq-no-pause-{uuid.uuid4().hex}")),
        )
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            sparkq._tick()
        return started

    def test_a_machine_that_cannot_see_the_gpu_starts_on_sessions_alone(self):
        """맥에서는 GPU를 못 읽어도(`None`) 큐가 나아간다.

        리눅스라면 `None`은 "확인하지 못했다"라 기다리는 것이 맞다. 맥에서는 영영 확인할
        수 없으므로 같은 규칙을 쓰면 큐가 한 번도 시작하지 않는다.
        """
        started = self._tick_with(True, mock.Mock(return_value=None))
        started.assert_not_called()

        started = self._tick_with(False, mock.Mock(return_value=None))
        started.assert_called_once()

    def test_gpu_apps_is_absent_rather_than_empty_when_it_cannot_be_read(self):
        """볼 수 없는 기계는 그 칸을 **싣지 않는다.**

        빈 목록은 "확인했고 비어 있다"로 읽힌다. 그 둘은 사람이 할 일이 정반대다.
        """
        patches = (
            mock.patch.object(sparkq, "current_job", return_value=None),
            mock.patch.object(sparkq, "current_side", return_value=None),
            mock.patch.object(sparkq, "queued_files", return_value=[]),
            mock.patch.object(sparkq, "recent", return_value=[]),
            mock.patch.object(sparkq, "train_sessions", return_value=[]),
            mock.patch.object(sparkq, "compute_apps", return_value=[]),
            mock.patch.object(sparkq, "PAUSED_FILE", Path(f"/tmp/sparkq-no-pause-{uuid.uuid4().hex}")),
        )
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(mock.patch.object(sparkq, "WATCHES_GPU_PROCESSES", True))
            watching = sparkq.snapshot()
            stack.enter_context(mock.patch.object(sparkq, "WATCHES_GPU_PROCESSES", False))
            blind = sparkq.snapshot()

        self.assertEqual(watching["gpu_apps"], [])
        self.assertNotIn("gpu_apps", blind)
        self.assertFalse(blind["capabilities"]["gpu_processes"])


class ResumeSourceTest(unittest.TestCase):
    def test_runs_lists_checkpointed_output_directories_newest_first(self):
        with tempfile.TemporaryDirectory() as raw:
            outputs = Path(raw)
            for name, step, when in (("older__act__aaaa", "004000", 1), ("newer__smolvla__bbbb", "010000", 2)):
                checkpoints = outputs / name / "checkpoints"
                (checkpoints / step / "pretrained_model").mkdir(parents=True)
                (checkpoints / step / "pretrained_model" / "train_config.json").write_text(json.dumps({
                    "steps": 20000,
                    "policy": {"type": name.split("__")[1]},
                    "dataset": {"repo_id": "soarm101_x"},
                }))
                (checkpoints / "last").symlink_to(step)
                import os
                os.utime(checkpoints / step / "pretrained_model" / "train_config.json", (when, when))
            # 체크포인트가 없는 것은 이어붙일 수 없으므로 목록에 없다.
            (outputs / "no_checkpoint").mkdir()

            with mock.patch.object(sparkq, "OUTPUT_ROOT", outputs):
                found = sparkq.runs()

        self.assertEqual([item["name"] for item in found], ["newer__smolvla__bbbb", "older__act__aaaa"])
        self.assertEqual(found[0]["step"], 10000)
        self.assertEqual(found[0]["steps"], 20000)
        self.assertEqual(found[0]["policy"], "smolvla")

    def test_a_name_that_is_not_in_its_source_is_refused_when_queued(self):
        """없는 것을 고르면 **걸 때** 막는다.

        통과시키면 그 작업은 새벽에 시작해 몇 초 만에 죽고 큐는 다음으로 넘어간다.
        아침에 남는 것은 실패 한 줄과 날아간 밤 하나다.
        """
        spec = {"fields": [{"name": "run", "type": "name", "source": "runs"}]}
        with mock.patch.object(sparkq, "runs", return_value=[{"name": "here__act__aaaa"}]):
            self.assertEqual(
                sparkq.validate(spec, {"run": "here__act__aaaa"}), {"run": "here__act__aaaa"}
            )
            with self.assertRaises(sparkq.Invalid):
                sparkq.validate(spec, {"run": "gone__act__bbbb"})


class HTTPErrorTest(unittest.TestCase):
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
