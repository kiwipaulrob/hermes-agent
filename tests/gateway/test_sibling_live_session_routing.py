"""Profile-less ancillary polls route to the live session's sibling authority.

Sibling authorities key ``profile_id`` by home PATH (``initialize_session_authority`` is
called with ``profile_id=str(home)``), while ``_sibling_for`` matches on the short profile
NAME via ``profile_matches_home``. ``_sibling_for_live_session`` used to hand the raw path
straight over, so ``PROFILE_ID_RE`` rejected it for every sibling and each ``session-routed``
marker was followed by ``profile_mismatch``. The handoff now resolves the path to its name
(and ``_sibling_for`` tolerates path-form input), so the routed verb succeeds.
"""
import logging
from pathlib import Path

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
from tests.gateway.test_ticket_profile_scope import _Transport


def _host_connection(runner, root):
    from gateway.session_authorities import authority_for_profile_id, owner_scope
    from gateway.session_controls import AuthorityConnection
    store = runner.session_ticket_store
    ticket = store.mint(profile_id=str(root.resolve()), subject='uid:desk', purpose='interactive', scope='host')
    grant = store.redeem(ticket, profile_id=None, purpose='interactive')
    launch = authority_for_profile_id(runner, grant['profile_id'])
    identity = {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
                'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
                'native_bootstrap': True, 'profile_scope': 'host'}
    with owner_scope(launch):
        return AuthorityConnection(launch, _Transport(), identity, operator=True), launch


@pytest.mark.asyncio
async def test_profile_less_poll_of_sibling_live_session_routes_and_succeeds(tmp_path, monkeypatch, caplog):
    """A ``session-routed`` marker is followed by a result, never by ``profile_mismatch``."""
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import owner_scope
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    beta_home = homes[2][1]
    connection, runner = None, None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        connection, launch = _host_connection(runner, root)

        async def create_on_beta():
            with owner_scope(launch):
                return await connection.dispatch({'id': 1, 'method': 'session.create', 'params': {
                    'profile': 'beta', 'request_id': 'regression-sid', 'source': 'cli',
                    'cwd': str(tmp_path), 'model': 'm', 'toolsets': []}})

        created = await create_on_beta()
        assert 'result' in created, created
        sid = created['result']['session_id']
        beta = runner.session_authorities.for_home(beta_home)
        assert sid in beta.sessions
        # The sibling keys profile_id by home path — the exact shape that used to crash.
        assert '/' in str(beta.profile_id)

        caplog.set_level(logging.INFO, logger='gateway.phone_diag')
        caplog.clear()
        with owner_scope(launch):
            for rid, method in ((2, 'subagent.list'), (3, 'session.control.read')):
                reply = await connection.dispatch(
                    {'id': rid, 'method': method, 'params': {'session_id': sid}})
                assert 'result' in reply, f'{method} on a live sibling session must route, not refuse: {reply}'
        routed = [r for r in caplog.records
                  if r.name == 'gateway.phone_diag' and 'HERMES-DIAG session-routed' in r.getMessage()]
        assert len(routed) == 2, [r.getMessage() for r in caplog.records if r.name == 'gateway.phone_diag']

        # Unknown sessions keep the local verdict (no crash, no sibling hop).
        with owner_scope(launch):
            missing = await connection.dispatch(
                {'id': 4, 'method': 'subagent.list', 'params': {'session_id': 'no-such-session'}})
        assert missing.get('error', {}).get('message') != 'profile_mismatch', missing
    finally:
        if connection is not None:
            with owner_scope(connection.authority):
                await connection.close()
        if runner is not None:
            for authority in list(getattr(runner, 'session_authorities', None) or []):
                authority.db.close()
        for _, home in homes:
            process_ownership.release(Path(home))


@pytest.mark.asyncio
async def test_sibling_for_accepts_path_form_profile_id(tmp_path, monkeypatch):
    """``_sibling_for`` resolves a stored home path to the name its gates match on."""
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import owner_scope
    from hermes_state_runtime import RuntimeStoreError
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    beta_home = homes[2][1]
    connection, runner = None, None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        connection, _launch = _host_connection(runner, root)
        _beta = runner.session_authorities.for_home(beta_home)
        assert _beta is not None
        assert connection._sibling_for(str(beta_home)) is not None
        assert connection._sibling_for('beta') is not None
        assert connection._sibling_for('default') is None
        assert connection._sibling_for(str(root.resolve())) is None
        with pytest.raises(RuntimeStoreError) as exc:
            connection._sibling_for('no-such-profile')
        assert exc.value.reason == 'profile_mismatch'
    finally:
        if connection is not None:
            with owner_scope(connection.authority):
                await connection.close()
        if runner is not None:
            for authority in list(getattr(runner, 'session_authorities', None) or []):
                authority.db.close()
        for _, home in homes:
            process_ownership.release(Path(home))
