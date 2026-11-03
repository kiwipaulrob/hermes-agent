"""prompt.submit adopts an entitled sender its socket never attached.

A phone that relogged or reconnected submits on a fresh socket with a runtime id
the gateway has no subscription for. The submitter is adopted when it is entitled
to submit; a stranger without the capability is still refused.
"""
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection, SessionRef
from tests.gateway.test_local_source_labels import _authority, _create


@pytest.mark.asyncio
async def test_submit_adopts_entitled_sender_on_fresh_socket(tmp_path, monkeypatch):
    authority, store = await _authority(tmp_path, monkeypatch)
    home = fresh = None
    try:
        home = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None),
                                   {'user_id': 'owner'})
        sid = await _create(home, authority, store, 'cli', 'adopt')
        assert sid in home.subscriptions
        # A new socket for the same owner: nothing attached, like the phone after
        # a relogin whose resume hasn't landed.
        fresh = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None),
                                    {'user_id': 'owner'})
        assert sid not in fresh.subscriptions
        await fresh._adopt_submitter(SessionRef(home.actor.profile_id, sid))
        assert sid in fresh.subscriptions
    finally:
        for conn in (home, fresh):
            if conn is not None:
                await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_submit_still_refuses_stranger_without_capability(tmp_path, monkeypatch):
    from hermes_state_runtime import RuntimeStoreError
    authority, store = await _authority(tmp_path, monkeypatch)
    home = stranger = None
    try:
        home = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None),
                                   {'user_id': 'owner'})
        sid = await _create(home, authority, store, 'cli', 'adopt-stranger')
        # No user_id: no capabilities at all.
        stranger = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {})
        with pytest.raises(RuntimeStoreError) as exc:
            await stranger._adopt_submitter(SessionRef(home.actor.profile_id, sid))
        assert exc.value.reason == 'permission_denied'
        assert sid not in stranger.subscriptions
    finally:
        for conn in (home, stranger):
            if conn is not None:
                await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_submit_truncation_refuses_unconfirmed_cut(tmp_path, monkeypatch):
    authority, store = await _authority(tmp_path, monkeypatch)
    home = None
    try:
        home = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None),
                                   {'user_id': 'owner'})
        sid = await _create(home, authority, store, 'cli', 'truncate-unconfirmed')
        row_id = authority.db.append_message(sid, 'user', 'first')
        from hermes_state_runtime import RuntimeStoreError
        with pytest.raises(RuntimeStoreError) as exc:
            await home._apply_submit_truncation(
                SessionRef(home.actor.profile_id, sid),
                {'session_id': sid, 'truncate_before_row_id': row_id, 'text': 'x'})
        assert exc.value.reason == 'invalid_params'
        # Nothing cut.
        rows = authority.db.get_messages_as_conversation(sid, include_row_ids=True)
        assert [r['_row_id'] for r in rows if r.get('role') == 'user'] == [row_id]
    finally:
        if home is not None:
            await home.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_submit_truncation_refuses_unknown_row(tmp_path, monkeypatch):
    from hermes_state_runtime import RuntimeStoreError
    authority, store = await _authority(tmp_path, monkeypatch)
    home = None
    try:
        home = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None),
                                   {'user_id': 'owner'})
        sid = await _create(home, authority, store, 'cli', 'truncate-unknown')
        authority.db.append_message(sid, 'user', 'first')
        with pytest.raises(RuntimeStoreError) as exc:
            await home._apply_submit_truncation(
                SessionRef(home.actor.profile_id, sid),
                {'session_id': sid, 'truncate_before_row_id': 999999,
                 'confirm_truncate': True, 'text': 'x'})
        assert exc.value.reason == 'invalid_params'
    finally:
        if home is not None:
            await home.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_submit_truncation_cuts_before_target_row(tmp_path, monkeypatch):
    authority, store = await _authority(tmp_path, monkeypatch)
    home = None
    try:
        home = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None),
                                   {'user_id': 'owner'})
        sid = await _create(home, authority, store, 'cli', 'truncate-cut')
        first = authority.db.append_message(sid, 'user', 'first')
        authority.db.append_message(sid, 'assistant', 'reply')
        second = authority.db.append_message(sid, 'user', 'second')
        await home._apply_submit_truncation(
            SessionRef(home.actor.profile_id, sid),
            {'session_id': sid, 'truncate_before_row_id': second,
             'confirm_truncate': True, 'text': 'replacement'})
        rows = authority.db.get_messages_as_conversation(sid, include_row_ids=True)
        assert [(r.get('role'), r.get('_row_id')) for r in rows] == [
            ('user', first), ('assistant', rows[1].get('_row_id'))]
        assert second not in [r.get('_row_id') for r in rows]
    finally:
        if home is not None:
            await home.close()
        authority.db.close()
