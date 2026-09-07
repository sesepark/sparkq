import contextlib
import io
import json
import re
import shutil
import signal
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

            # 실행 목록은 이제 "지금 누가 이 실행을 쓰고 있는가"도 함께 답한다. 그
            # 물음은 진짜 큐와 tmux를 읽으므로, 이 시험에서는 비어 있는 기계로 둔다.
            with mock.patch.object(sparkq, "OUTPUT_ROOT", outputs), \
                    mock.patch.object(sparkq, "job_claims", return_value={}), \
                    mock.patch.object(sparkq, "train_sessions", return_value=[]):
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


class RunArtifactTest(unittest.TestCase):
    """학습이 남긴 것을 세고 지우는 길."""

    def make_run(self, outputs, name, steps, *, model=100, state=300):
        """체크포인트 몇 개짜리 실행 하나. `last`는 마지막 것을 가리킨다."""
        checkpoints = outputs / name / "checkpoints"
        for step in steps:
            pretrained = checkpoints / step / "pretrained_model"
            pretrained.mkdir(parents=True)
            (pretrained / "model.safetensors").write_bytes(b"m" * model)
            (pretrained / "train_config.json").write_text(json.dumps({
                "steps": 20000, "policy": {"type": "smolvla"}, "dataset": {"repo_id": "soarm101_x"},
            }))
            training_state = checkpoints / step / "training_state"
            training_state.mkdir()
            (training_state / "optimizer.safetensors").write_bytes(b"s" * state)
        (checkpoints / "last").symlink_to(steps[-1])
        return outputs / name

    @contextlib.contextmanager
    def machine(self, outputs, queued=(), current=None):
        """큐가 비어 있는(또는 지정한 것만 든) 기계 하나."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            queue_dir = root / "queue"
            queue_dir.mkdir()
            for index, job in enumerate(queued):
                (queue_dir / f"5000-{index:019d}-{job['id']}.json").write_text(json.dumps(job))
            (root / "current").write_text(current or "")
            if current:
                (root / "runs" / current).mkdir(parents=True)
                (root / "runs" / current / "job.json").write_text(
                    json.dumps({"id": current, "session": f"train-{current}"})
                )
            with mock.patch.object(sparkq, "OUTPUT_ROOT", outputs), \
                    mock.patch.object(sparkq, "ROOT", root), \
                    mock.patch.object(sparkq, "QUEUE_DIR", queue_dir), \
                    mock.patch.object(sparkq, "RUNS_DIR", root / "runs"), \
                    mock.patch.object(sparkq, "train_sessions", return_value=[]), \
                    mock.patch.object(sparkq, "session_alive", return_value=False):
                yield

    def test_optimizer_state_is_counted_apart_from_the_weights(self):
        """합쳐 놓으면 화면이 "무엇을 지우면 무엇을 잃는가"를 말할 수 없다."""
        with tempfile.TemporaryDirectory() as raw:
            outputs = Path(raw)
            directory = self.make_run(outputs, "a__smolvla__aaaa", ["005000", "010000"])

            found = sparkq.checkpoints_of(directory)

        # `last`는 심볼릭 링크다. 따라가면 같은 것이 두 번 세어진다.
        self.assertEqual([item["step"] for item in found], ["005000", "010000"])
        self.assertEqual(found[0]["state_bytes"], 300)
        self.assertGreater(found[0]["model_bytes"], 100)
        self.assertEqual(found[0]["bytes"], found[0]["model_bytes"] + 300)

    def test_deleting_a_checkpoint_does_not_hide_the_whole_run(self):
        """`last`가 끊어지면 실행 전체가 목록에서 사라진다 — 지운 것은 하나였는데."""
        with tempfile.TemporaryDirectory() as raw:
            outputs = Path(raw)
            self.make_run(outputs, "a__smolvla__aaaa", ["005000", "010000"])

            with self.machine(outputs):
                sparkq.delete_checkpoint("a__smolvla__aaaa", "010000")
                found = sparkq.runs()

            link = outputs / "a__smolvla__aaaa" / "checkpoints" / "last"
            self.assertEqual(link.resolve().name, "005000")
        self.assertEqual([item["name"] for item in found], ["a__smolvla__aaaa"])
        self.assertEqual(found[0]["step"], 5000)

    def test_dropping_only_the_optimizer_state_keeps_the_weights(self):
        with tempfile.TemporaryDirectory() as raw:
            outputs = Path(raw)
            self.make_run(outputs, "a__smolvla__aaaa", ["005000"])

            with self.machine(outputs):
                result = sparkq.delete_checkpoint("a__smolvla__aaaa", "005000", only_state=True)

            step = outputs / "a__smolvla__aaaa" / "checkpoints" / "005000"
            self.assertTrue((step / "pretrained_model").is_dir())
            self.assertFalse((step / "training_state").exists())
        self.assertEqual(result["freed_bytes"], 300)
        self.assertFalse(result["resumable"])

    def test_a_run_a_waiting_job_points_at_is_not_deleted(self):
        """대기 중인 `lerobot-resume`이 가리키는 실행을 지우면 새벽에 죽는다."""
        with tempfile.TemporaryDirectory() as raw:
            outputs = Path(raw)
            self.make_run(outputs, "a__smolvla__aaaa", ["005000"])
            waiting = {"id": "20260906_1-0001", "session": "train-a__smolvla__aaaa"}

            with self.machine(outputs, queued=[waiting]):
                with self.assertRaises(sparkq.Conflict):
                    sparkq.delete_run("a__smolvla__aaaa")
                listed = sparkq.runs()

            self.assertTrue((outputs / "a__smolvla__aaaa").is_dir())
        self.assertIn("20260906_1-0001", listed[0]["in_use"])

    def test_a_run_folder_pointing_outside_outputs_is_refused(self):
        """이름 검사를 통과해도 심볼릭 링크가 바깥을 가리킬 수 있다."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outputs = root / "outputs"
            outputs.mkdir()
            elsewhere = root / "elsewhere"
            (elsewhere / "checkpoints").mkdir(parents=True)
            (outputs / "escape").symlink_to(elsewhere)

            with self.machine(outputs):
                with self.assertRaises(sparkq.Invalid):
                    sparkq.delete_run("escape")

            self.assertTrue(elsewhere.is_dir())


class StopGraceTest(unittest.TestCase):
    """중지 신호를 준 뒤 작업이 스스로 정리할 시간을 준다.

    전에는 2초를 자고 무조건 세션을 죽였다. Isaac 뷰어의 정리는 컨테이너 안에
    `pkill -INT`를 보내고 5초를 기다렸다가 `pkill -KILL`을 보내는데, 그 `sleep 5`
    도중에 세션이 죽어 KILL이 영영 실행되지 않았다 — 남은 `play.py`가 GPU를 쥐면 큐는
    다음 작업을 꺼내지 못한다.
    """

    def run_end_session(self, alive_sequence, grace=2.0):
        """`session_alive`가 이 순서대로 답할 때 tmux에 무엇을 보냈는지."""
        calls = []
        answers = list(alive_sequence)

        def fake_alive(_name):
            return answers.pop(0) if answers else False

        def fake_run(args, **_kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        with mock.patch.object(sparkq, "session_alive", side_effect=fake_alive), \
                mock.patch.object(sparkq, "stop_grace_of", return_value=grace), \
                mock.patch.object(sparkq.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(sparkq.time, "sleep"):
            sparkq.end_session({"kind": "isaac-play", "session": "side-viewer-1"})
        return calls

    def test_a_session_that_cleans_itself_up_is_never_killed(self):
        # 처음엔 살아 있고(정리 중), 곧 스스로 사라진다.
        calls = self.run_end_session([True, True, False])

        self.assertEqual(calls[0][:2], ["tmux", "send-keys"])
        self.assertNotIn("kill-session", [arg for call in calls for arg in call])

    def test_a_session_that_never_leaves_is_killed_after_the_grace(self):
        calls = self.run_end_session([True] * 50, grace=0.0)

        self.assertIn("kill-session", [arg for call in calls for arg in call])

    def test_the_grace_comes_from_the_kind_and_is_capped(self):
        with mock.patch.object(sparkq, "load_kinds", return_value={"slow": {"stop_grace_seconds": 9999}}):
            self.assertEqual(sparkq.stop_grace_of({"kind": "slow"}), sparkq.MAX_STOP_GRACE)
        with mock.patch.object(sparkq, "load_kinds", return_value={"plain": {}}):
            self.assertEqual(sparkq.stop_grace_of({"kind": "plain"}), sparkq.DEFAULT_STOP_GRACE)
        with mock.patch.object(sparkq, "load_kinds", return_value={"bad": {"stop_grace_seconds": "곧"}}):
            self.assertEqual(sparkq.stop_grace_of({"kind": "bad"}), sparkq.DEFAULT_STOP_GRACE)

    def test_every_kind_waits_longer_than_its_own_cleanup(self):
        """종류 파일과 그 안의 정리 스크립트가 서로 맞아야 한다.

        정리가 `sleep`을 쓰면 그 시간이 유예 안에 들어와야 한다. 안 그러면 정리가 중간에
        끊기고, 그것이 바로 컨테이너 안에 프로세스를 남기는 길이다. 대부분의 종류는
        기본값으로 충분하므로 `stop_grace_seconds`를 적지 않는다 — 적지 않은 것까지
        함께 본다.
        """
        for path in sorted((Path(__file__).parent / "kinds").glob("*.json")):
            with self.subTest(kind=path.name):
                spec = json.loads(path.read_text())
                with mock.patch.object(sparkq, "load_kinds", return_value={spec["kind"]: spec}):
                    grace = sparkq.stop_grace_of({"kind": spec["kind"]})
                sleeps = [int(n) for n in re.findall(r"sleep (\d+)", spec.get("run", ""))]

                self.assertLessEqual(sum(sleeps) + 2, grace)


class RslMetricsTest(unittest.TestCase):
    """rsl_rl은 반복마다 서른 줄이 넘는 지표를 찍는다. 지금까지 하나만 보고 있었다."""

    BLOCK = """
                          Learning iteration 480/3000
                            Total steps: 47284224
                       Steps per second: 58112
                        Collection time: 1.572s
                        Mean value loss: 3.6246
                    Mean surrogate loss: -0.0023
                            Mean reward: 155.18
                   Metrics/success_rate: 0.0000
         Episode_Reward/reaching_object: 0.0045
          Episode_Reward/lifting_object: 10.4019
                 Curriculum/grasp_stage: 2.0000
           Episode_Termination/time_out: 0.9979
--------------------------------------------------------------------------------
                         Iteration time: 1.69s
                           Time elapsed: 00:13:52
                                    ETA: 01:12:41
"""

    def series_of(self, iterations=(480, 481)):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "runs" / "job1").mkdir(parents=True)
            (root / "runs" / "job1" / "job.json").write_text(json.dumps({"progress": "rsl_rl"}))
            log = "".join(self.BLOCK.replace("iteration 480/", f"iteration {n}/") for n in iterations)
            (root / "runs" / "job1" / "run.log").write_text(log)
            with mock.patch.object(sparkq, "RUNS_DIR", root / "runs"):
                return sparkq.series("job1")

    def test_every_number_in_the_block_becomes_a_curve(self):
        found = self.series_of()
        names = [s["name"] for s in found["series"]]

        # 평균 보상은 첫째로 남는다 — 지금까지 화면이 그리던 것이다.
        self.assertEqual(names[0], "reward")
        for expected in ("Mean value loss", "Metrics/success_rate",
                         "Episode_Reward/lifting_object", "Curriculum/grasp_stage",
                         "Episode_Termination/time_out", "Steps per second"):
            self.assertIn(expected, names)

    def test_clock_shaped_lines_are_not_curves(self):
        """`ETA: 01:12:41`과 `Time elapsed`는 숫자가 아니라 시계다."""
        names = [s["name"] for s in self.series_of()["series"]]

        self.assertNotIn("ETA", names)
        self.assertNotIn("Time elapsed", names)
        self.assertNotIn("Learning iteration", names)

    def test_curves_are_grouped_and_named_for_reading(self):
        found = {s["name"]: s for s in self.series_of()["series"]}

        self.assertEqual(found["Episode_Reward/lifting_object"]["group"], "보상 항목")
        self.assertEqual(found["Episode_Reward/lifting_object"]["label"], "lifting object")
        self.assertEqual(found["Mean value loss"]["group"], "손실")
        self.assertEqual(found["Mean value loss"]["label"], "가치 손실")
        self.assertEqual(found["Curriculum/grasp_stage"]["group"], "커리큘럼")
        self.assertEqual(found["Steps per second"]["group"], "속도")

    def test_a_single_point_is_not_a_curve(self):
        """한 반복만 돈 학습에서 점 하나짜리 곡선을 서른 개 그리지 않는다."""
        found = self.series_of(iterations=(480,))

        self.assertEqual([s["name"] for s in found["series"]], ["reward"])


class OwnershipTest(unittest.TestCase):
    """누구의 GPU 프로세스인가. 멈춰 두기가 이 판단 위에 서 있다."""

    def test_container_process_of_the_running_job_is_not_foreign(self):
        """`docker exec`로 도는 자기 학습을 남의 것으로 세지 않는다.

        2026-09-07에 실제로 그랬다. Isaac 학습의 GPU 프로세스는 tmux pane의 자손이 아니라
        containerd-shim의 자손이라 세션 트리 검사를 빠져나갔고, 앱은 큐가 방금 띄운 학습을
        "큐 밖의 프로세스"로 보여 주고 있었다.
        """
        job = {"id": "training", "session": "train-x"}
        apps = [{"pid": "999"}]
        with mock.patch.object(sparkq, "session_process_ids", return_value={"100", "101"}), \
                mock.patch.object(sparkq, "session_containers", return_value={"c" * 64}), \
                mock.patch.object(sparkq, "container_of", return_value="c" * 64):
            self.assertEqual(sparkq.job_gpu_pids(job, apps), ["999"])
            self.assertEqual(sparkq.foreign_apps(apps, job, None), [])

    def test_a_process_in_another_container_stays_foreign(self):
        job = {"id": "training", "session": "train-x"}
        apps = [{"pid": "999"}]
        with mock.patch.object(sparkq, "session_process_ids", return_value={"100"}), \
                mock.patch.object(sparkq, "session_containers", return_value={"a" * 64}), \
                mock.patch.object(sparkq, "container_of", return_value="b" * 64):
            self.assertEqual(sparkq.job_gpu_pids(job, apps), [])
            self.assertEqual(sparkq.foreign_apps(apps, job, None), apps)


class PreemptTest(unittest.TestCase):
    """멈춰 두기. 깨우는 쪽이 실패하면 학습이 영원히 얼어붙으므로 그쪽을 더 많이 시험한다."""

    def kinds_dir(self, stack, preempt=True):
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        (directory / "viewer.json").write_text(json.dumps({
            "kind": "viewer", "lane": "side", "limit_seconds": 600, "preempt": preempt,
            "title": "Viewer", "session": "side-viewer-${id}", "fields": [], "run": "true",
        }))
        return directory

    def test_preempt_is_refused_on_a_queue_kind(self):
        """큐 종류는 학습을 멈추겠다고 말할 수 없다 — 깨워 줄 시한이 없기 때문이다."""
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "bad.json").write_text(json.dumps({"kind": "bad", "preempt": True, "run": "true"}))
            (directory / "good.json").write_text(json.dumps({
                "kind": "good", "lane": "side", "limit_seconds": 600, "preempt": True, "run": "true",
            }))
            stderr = io.StringIO()
            with mock.patch.object(sparkq, "KINDS_DIR", directory), contextlib.redirect_stderr(stderr):
                kinds = sparkq.load_kinds()

        self.assertEqual(set(kinds), {"good"})
        self.assertIn("bad.json", stderr.getvalue())

    def test_nothing_to_freeze_is_not_an_error(self):
        with mock.patch.object(sparkq, "current_job", return_value=None):
            self.assertIsNone(sparkq.preempt_for({"id": "side"}))

    def test_a_training_whose_gpu_process_is_unknown_is_not_preempted(self):
        """멈출 대상을 모르면 곁다리를 시작하지 않는다.

        모른 채로 옆에서 추론을 띄우면 둘이 GPU를 나눠 쓴다 — 이 기능이 막으려던 상태다.
        """
        with mock.patch.object(sparkq, "current_job", return_value={"id": "t", "session": "train-t"}), \
                mock.patch.object(sparkq, "session_alive", return_value=True), \
                mock.patch.object(sparkq, "preempt_targets", return_value=[]):
            with self.assertRaises(sparkq.Conflict):
                sparkq.preempt_for({"id": "side"})

    def test_freeze_and_wake_are_recorded_on_disk(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runs = root / "runs" / "t"
            runs.mkdir(parents=True)
            (runs / "job.json").write_text(json.dumps({"id": "t", "session": "train-t"}))
            signals = []
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(sparkq, "ROOT", root))
                stack.enter_context(mock.patch.object(sparkq, "RUNS_DIR", root / "runs"))
                stack.enter_context(mock.patch.object(sparkq, "PREEMPTED_FILE", root / "preempted"))
                stack.enter_context(mock.patch.object(
                    sparkq, "current_job", return_value={"id": "t", "session": "train-t"}))
                stack.enter_context(mock.patch.object(sparkq, "session_alive", return_value=True))
                stack.enter_context(mock.patch.object(sparkq, "preempt_targets", return_value=["7", "9"]))
                stack.enter_context(mock.patch.object(
                    sparkq.os, "kill", side_effect=lambda pid, number: signals.append((pid, number))))

                record = sparkq.preempt_for({"id": "side-1", "kind": "viewer"})
                self.assertEqual(record["pids"], ["7", "9"])
                self.assertTrue((root / "preempted").exists())
                self.assertEqual([number for _, number in signals], [signal.SIGSTOP] * 2)

                woken = sparkq.resume_preempted()
                self.assertEqual([number for _, number in signals[2:]], [signal.SIGCONT] * 2)
                self.assertFalse((root / "preempted").exists())
                self.assertGreaterEqual(woken["paused_seconds"], 0.0)
                self.assertIn("paused_seconds", json.loads((runs / "job.json").read_text()))

                # 두 번 깨워도 안전하다. 깨우는 손이 여럿(곁다리의 trap·데몬·박자)이라
                # 이 성질이 없으면 그 가운데 하나가 다른 하나를 깨뜨린다.
                self.assertIsNone(sparkq.resume_preempted())

    def test_side_start_freezes_training_and_leaves_a_trap_that_wakes_it(self):
        with contextlib.ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            kinds = self.kinds_dir(stack)
            stack.enter_context(mock.patch.object(sparkq, "ROOT", root))
            stack.enter_context(mock.patch.object(sparkq, "RUNS_DIR", root / "runs"))
            stack.enter_context(mock.patch.object(sparkq, "CURRENT_SIDE_FILE", root / "current_side"))
            stack.enter_context(mock.patch.object(sparkq, "PREEMPTED_FILE", root / "preempted"))
            stack.enter_context(mock.patch.object(sparkq, "KINDS_DIR", kinds))
            stack.enter_context(mock.patch.object(
                sparkq, "current_job", return_value={"id": "t", "session": "train-t"}))
            stack.enter_context(mock.patch.object(sparkq, "session_alive", return_value=True))
            stack.enter_context(mock.patch.object(sparkq, "preempt_targets", return_value=["7"]))
            stack.enter_context(mock.patch.object(sparkq, "progress_of", return_value={}))
            stack.enter_context(mock.patch.object(sparkq.os, "kill"))
            stack.enter_context(mock.patch.object(sparkq.subprocess, "run"))
            stack.enter_context(mock.patch.object(
                sparkq.probe, "timeout_prefix", return_value="timeout --signal=INT 3600"))

            side = sparkq.start_side("viewer", {})
            job_id = (root / "current_side").read_text().strip()
            wrapper = (root / "runs" / job_id / "run.sh").read_text()

        self.assertEqual(side["preempted_job"], "t")
        # `exec`이면 trap을 실행할 셸이 남지 않는다. 깨우는 두 번째 손이 사라지는 자리다.
        self.assertIn("trap 'kill -CONT 7 2>/dev/null' EXIT INT TERM", wrapper)
        self.assertNotIn("exec timeout", wrapper)

    def test_a_side_that_never_started_does_not_leave_a_frozen_training(self):
        with contextlib.ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            kinds = self.kinds_dir(stack)
            stack.enter_context(mock.patch.object(sparkq, "ROOT", root))
            stack.enter_context(mock.patch.object(sparkq, "RUNS_DIR", root / "runs"))
            stack.enter_context(mock.patch.object(sparkq, "CURRENT_SIDE_FILE", root / "current_side"))
            stack.enter_context(mock.patch.object(sparkq, "PREEMPTED_FILE", root / "preempted"))
            stack.enter_context(mock.patch.object(sparkq, "KINDS_DIR", kinds))
            stack.enter_context(mock.patch.object(
                sparkq, "current_job", return_value={"id": "t", "session": "train-t"}))
            stack.enter_context(mock.patch.object(sparkq, "session_alive", return_value=True))
            stack.enter_context(mock.patch.object(sparkq, "preempt_targets", return_value=["7"]))
            stack.enter_context(mock.patch.object(sparkq, "progress_of", return_value={}))
            stack.enter_context(mock.patch.object(sparkq.os, "kill"))
            stack.enter_context(mock.patch.object(
                sparkq.probe, "timeout_prefix", return_value="timeout --signal=INT 3600"))
            stack.enter_context(mock.patch.object(
                sparkq.subprocess, "run", side_effect=OSError("tmux가 없습니다")))

            with self.assertRaises(OSError):
                sparkq.start_side("viewer", {})

            self.assertFalse((root / "preempted").exists())

    def test_stopping_a_training_stops_the_viewer_that_follows_it(self):
        """학습이 서면 그 학습에 딸린 곁다리(뷰어)도 함께 선다. 다른 학습의 것, 학습이 끝난 뒤
        띄운 것(`follows` 없음)은 그대로 둔다."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "runs" / "viewer").mkdir(parents=True)
            follower = {"id": "viewer", "session": "side-viewer", "follows": "t"}
            for side, stops in (
                (follower, 1),
                ({"id": "viewer", "session": "side-viewer", "follows": "other"}, 0),
                ({"id": "viewer", "session": "side-viewer"}, 0),
                (None, 0),
            ):
                with mock.patch.object(sparkq, "RUNS_DIR", root / "runs"), \
                        mock.patch.object(sparkq, "current_side", return_value=side), \
                        mock.patch.object(sparkq, "_stop_side") as stop_side:
                    sparkq.stop_following_side("t")
                self.assertEqual(stop_side.call_count, stops, side)
            note = json.loads((root / "runs" / "viewer" / "job.json").read_text())
            self.assertIn("t", note["note"])


class QueueOrderTest(unittest.TestCase):
    """줄 순서를 바꾸는 것. **파일 이름 하나가 곧 순서다**(`queue_path`)."""

    @contextlib.contextmanager
    def queue(self, ids, priorities=None):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            queue_dir = root / "queue"
            queue_dir.mkdir()
            for index, job_id in enumerate(ids):
                priority = 5000 if priorities is None else priorities[index]
                created = 1_700_000_000 + index
                (queue_dir / f"{priority:04d}-{created * 10**9:019d}-{job_id}.json").write_text(
                    json.dumps({
                        "id": job_id, "kind": "lerobot-train", "title": job_id,
                        "priority": priority, "created_at": float(created), "state": "queued",
                    }),
                    encoding="utf-8",
                )
            with mock.patch.object(sparkq, "ROOT", root), \
                    mock.patch.object(sparkq, "QUEUE_DIR", queue_dir), \
                    mock.patch.object(sparkq, "RUNS_DIR", root / "runs"):
                yield

    def order(self):
        return [job["id"] for job in sparkq.queued_jobs()]

    def test_moving_one_before_another_puts_it_exactly_there(self):
        with self.queue(["a", "b", "c", "d"]):
            sparkq.move("d", "b")

            self.assertEqual(self.order(), ["a", "d", "b", "c"])

    def test_moving_without_a_target_sends_it_to_the_back(self):
        with self.queue(["a", "b", "c"]):
            sparkq.move("a", None)

            self.assertEqual(self.order(), ["b", "c", "a"])

    def test_moving_up_one_step_is_moving_before_the_one_above(self):
        """화면의 `위로`가 이 한 가지로 표현된다 — 자리를 번호가 아니라 이름으로 말한다."""
        with self.queue(["a", "b", "c"]):
            sparkq.move("c", "b")

            self.assertEqual(self.order(), ["a", "c", "b"])

    def test_the_queue_never_keeps_a_ghost_of_the_moved_job(self):
        """쓰기와 지우기 사이에 같은 작업이 두 이름으로 남으면 줄에 유령이 선다."""
        with self.queue(["a", "b", "c"]):
            sparkq.move("c", "a")

            self.assertEqual(len(sparkq.queued_files()), 3)
            self.assertEqual(sorted(self.order()), ["a", "b", "c"])

    def test_priorities_are_renumbered_so_top_never_hits_the_floor(self):
        """예전 `top`은 `lowest - 1`로 내려가다 0에서 멈췄고, 그 뒤로는 앞으로 가지 않았다."""
        with self.queue(["a", "b"], priorities=[0, 0]):
            sparkq.move_to_top("b")

            self.assertEqual(self.order(), ["b", "a"])
            self.assertEqual([job["priority"] for job in sparkq.queued_jobs()], [0, 1])

    def test_a_newly_queued_job_still_lines_up_behind_the_reordered_ones(self):
        """줄에 선 것들이 0..N-1을 쓰므로, 기본 우선순위(5000)는 늘 뒤다."""
        with self.queue(["a", "b"]):
            sparkq.move("b", "a")
            queue_dir = sparkq.QUEUE_DIR
            (queue_dir / f"5000-{1_800_000_000 * 10**9:019d}-z.json").write_text(
                json.dumps({"id": "z", "priority": 5000, "created_at": 1_800_000_000.0}),
                encoding="utf-8",
            )

            self.assertEqual(self.order(), ["b", "a", "z"])

    def test_an_unknown_job_is_refused_on_both_sides(self):
        with self.queue(["a", "b"]):
            with self.assertRaises(sparkq.Missing):
                sparkq.move("zzz", "a")
            with self.assertRaises(sparkq.Missing):
                sparkq.move("a", "zzz")
            with self.assertRaises(sparkq.Invalid):
                sparkq.move("a", "a")

            self.assertEqual(self.order(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
