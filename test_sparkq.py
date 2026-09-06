import shutil
import unittest
import uuid

import sparkq


class TrainSessionsTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("tmux"), "tmux가 설치되어 있어야 합니다")
    def test_missing_tmux_server_is_an_empty_session_list(self):
        socket_name = f"sparkq-test-{uuid.uuid4().hex}"

        self.assertEqual(sparkq.train_sessions(tmux_socket=socket_name), [])


if __name__ == "__main__":
    unittest.main()
