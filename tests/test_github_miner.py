import random
import unittest

import httpx

from github_miner import GitHubMiner, GitHubTokenRotator, clear_recent_events_cache


class FakeGitHubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers_seen = []

    def get(self, path, params=None, headers=None):
        self.headers_seen.append(dict(headers or {}))
        response = self.responses.pop(0)
        response.request = httpx.Request("GET", "https://api.github.com" + path)
        return response


class GitHubTokenRotatorTest(unittest.TestCase):
    def tearDown(self):
        clear_recent_events_cache()

    def test_401_disables_token_and_retries_next_token(self):
        rotator = GitHubTokenRotator(["bad-token", "good-token"])
        miner = GitHubMiner(token_rotator=rotator, rng=random.Random(1))
        miner._client = FakeGitHubClient([
            httpx.Response(401, json={"message": "Bad credentials"}),
            httpx.Response(200, json={"ok": True}),
        ])

        self.assertEqual(miner._get_json("/events"), {"ok": True})
        self.assertEqual(rotator.active_count, 1)
        self.assertEqual(
            miner._client.headers_seen,
            [
                {"Authorization": "Bearer bad-token"},
                {"Authorization": "Bearer good-token"},
            ],
        )

    def test_all_401_tokens_fall_back_to_unauthenticated_request(self):
        rotator = GitHubTokenRotator(["bad-token"])
        miner = GitHubMiner(token_rotator=rotator, rng=random.Random(1))
        miner._client = FakeGitHubClient([
            httpx.Response(401, json={"message": "Bad credentials"}),
            httpx.Response(200, json={"ok": True}),
        ])

        self.assertEqual(miner._get_json("/events"), {"ok": True})
        self.assertEqual(rotator.active_count, 0)
        self.assertEqual(
            miner._client.headers_seen,
            [
                {"Authorization": "Bearer bad-token"},
                {},
            ],
        )

    def test_sample_commit_reuses_events_across_attempts(self):
        miner = GitHubMiner(rng=random.Random(1))
        events = [
            {
                "type": "PushEvent",
                "id": "event-1",
                "repo": {"name": "owner/repo"},
                "payload": {"commits": [{"sha": "bad-sha"}]},
            }
        ]
        calls = {"events": 0, "commit": 0}

        def fake_recent_events():
            calls["events"] += 1
            return events

        def fake_fetch_commit_candidate(**_kwargs):
            calls["commit"] += 1
            raise ValueError("bad commit")

        miner._recent_push_events = fake_recent_events
        miner._fetch_commit_candidate = fake_fetch_commit_candidate

        with self.assertRaisesRegex(RuntimeError, "bad commit"):
            miner.sample_commit(max_attempts=3)

        self.assertEqual(calls, {"events": 1, "commit": 3})

    def test_recent_push_events_cache_is_shared_between_miners(self):
        event_payload = [
            {
                "type": "PushEvent",
                "id": "event-1",
                "repo": {"name": "owner/repo"},
                "payload": {"commits": [{"sha": "abc"}]},
            }
        ]
        response = httpx.Response(200, json=event_payload)
        response.headers["link"] = ""
        miner_a = GitHubMiner(rng=random.Random(1))
        miner_b = GitHubMiner(rng=random.Random(2))
        miner_a._client = FakeGitHubClient([response])
        miner_b._client = FakeGitHubClient([])

        self.assertEqual(miner_a._recent_push_events(), event_payload)
        self.assertEqual(miner_b._recent_push_events(), event_payload)
        self.assertEqual(len(miner_a._client.headers_seen), 1)
        self.assertEqual(len(miner_b._client.headers_seen), 0)


if __name__ == "__main__":
    unittest.main()
