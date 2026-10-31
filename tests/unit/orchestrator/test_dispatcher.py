import asyncio
import contextlib
import uuid
from types import SimpleNamespace

from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.types import DispatchFailure, InflightEpisode, Policy, Progress, TaskRequest


def test_train_group_budget_bounds_fresh_groups():
    source = SimpleNamespace(next_task=lambda step: TaskRequest(env_name="test", task=None, step=step))
    envs = SimpleNamespace(get=lambda name: SimpleNamespace(config=SimpleNamespace(group_size=1)))
    dispatcher = SimpleNamespace(
        train_source=source, progress=Progress(), policy=Policy(model_name="m"), train_group_budget=2
    )
    opened = [Dispatcher.next_fresh_group(dispatcher, "train", envs) for _ in range(3)]
    assert [group is not None for group in opened] == [True, True, False]
    assert dispatcher.train_group_budget == 0

    dispatcher.train_group_budget = None
    assert all(Dispatcher.next_fresh_group(dispatcher, "train", envs) is not None for _ in range(3))


def test_request_cancelled_from_inside_fails_its_group_member():
    async def run() -> list:
        task = asyncio.ensure_future(asyncio.sleep(10))
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        meta = InflightEpisode(
            kind="train",
            env_name="test",
            group_id=uuid.uuid4(),
            task=SimpleNamespace(key="k", hash="h"),
            policy_version=0,
            step=1,
        )
        out_q = asyncio.Queue()
        dispatcher = SimpleNamespace(
            inflight={task: meta},
            retire=lambda meta: None,
            release=lambda refund_admission: None,
            groups={},
            metrics=SimpleNamespace(record_error=lambda kind, env_name: None),
            complete_group_member=lambda meta, group: 0,
            out_q=out_q,
        )
        await Dispatcher.handle_completed_request(dispatcher, task)
        return [out_q.get_nowait() for _ in range(out_q.qsize())]

    (result,) = asyncio.run(run())
    assert isinstance(result, DispatchFailure)
    assert result.error.type == "CancelledError"
