"""Authentication recovery regressions; all external I/O is mocked."""
import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady

import custom_components.stellantis_vehicles as integration
from custom_components.stellantis_vehicles import config_flow, stellantis
from custom_components.stellantis_vehicles.const import (
    FIELD_ANONYMIZE_LOGS, FIELD_NOTIFICATIONS, FIELD_OAUTH_CODE_URL,
    FIELD_RECONFIGURE, FIELD_REMOTE_COMMANDS,
)
from custom_components.stellantis_vehicles.exceptions import CommunicationError
from custom_components.stellantis_vehicles.utils import get_datetime


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['scheduled_tokens_refresh', 'get_user_vehicles'])
@pytest.mark.parametrize('error', [ConfigEntryAuthFailed, CommunicationError, asyncio.CancelledError])
async def test_setup_cleans_up_auth_network_and_cancelled_failures(stage, error):
    client = MagicMock()
    client.scheduled_tokens_refresh = AsyncMock()
    client.get_user_vehicles = AsyncMock()
    client.async_shutdown = AsyncMock()
    getattr(client, stage).side_effect = error('synthetic failure')
    entry = SimpleNamespace(data={})
    expected = ConfigEntryNotReady if error is CommunicationError else error
    with patch.object(integration, 'StellantisVehicles', return_value=client):
        with pytest.raises(expected):
            await integration.async_setup_entry(Mock(), entry)
    client.async_shutdown.assert_awaited_once()
    assert entry.runtime_data is None
    if stage == 'scheduled_tokens_refresh':
        client.get_user_vehicles.assert_not_awaited()


def client_with_expired_token():
    client = stellantis.StellantisVehicles(Mock())
    client.save_config({'oauth': {'access_token': 'fake-access', 'refresh_token': 'fake-refresh',
                                'expires_in': (get_datetime() - timedelta(minutes=1)).isoformat()}})
    client._entry = SimpleNamespace(async_start_reauth=Mock())
    client.refresh_oauth_token_request = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_startup_invalid_grant_propagates_without_timer_or_vehicle_poll():
    client = client_with_expired_token()
    client.refresh_oauth_token_request.side_effect = ConfigEntryAuthFailed('invalid_grant')
    client.scheduled_mqtt_token_refresh = AsyncMock()
    with patch.object(stellantis, 'async_track_point_in_time') as timer:
        with pytest.raises(ConfigEntryAuthFailed):
            await client.scheduled_tokens_refresh()
    timer.assert_not_called()
    client._entry.async_start_reauth.assert_not_called()  # HA setup owns reauth.
    client.scheduled_mqtt_token_refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_invalid_grant_stops_old_timer():
    client = client_with_expired_token()
    cancel = Mock()
    client._oauth_token_scheduled = cancel
    client.refresh_oauth_token_request.side_effect = ConfigEntryAuthFailed('invalid_grant')
    with patch.object(stellantis, 'async_track_point_in_time') as timer:
        await client.scheduled_oauth_token_refresh(get_datetime())
    cancel.assert_called_once()
    assert client._oauth_token_scheduled is None
    timer.assert_not_called()
    client._entry.async_start_reauth.assert_called_once_with(client._hass)
    assert client._oauth_auth_failed


@pytest.mark.asyncio
async def test_transient_failure_retries_in_background_but_propagates_at_setup():
    client = client_with_expired_token()
    client.refresh_oauth_token_request.side_effect = CommunicationError('offline')
    with patch.object(stellantis, 'async_track_point_in_time') as timer:
        await client.scheduled_oauth_token_refresh(get_datetime())
        assert timer.call_args.args[2] > get_datetime()
        timer.reset_mock()
        with pytest.raises(CommunicationError):
            await client.scheduled_oauth_token_refresh(setup=True)
        timer.assert_not_called()
    client._entry.async_start_reauth.assert_not_called()


@pytest.mark.asyncio
async def test_short_lived_token_does_not_schedule_in_the_past():
    client = client_with_expired_token()
    client._config['oauth']['expires_in'] = (get_datetime() + timedelta(seconds=60)).isoformat()
    with patch.object(stellantis, 'async_track_point_in_time') as timer:
        await client.scheduled_oauth_token_refresh()
        assert timer.call_args.args[2] >= get_datetime() + timedelta(seconds=29)


@pytest.mark.asyncio
async def test_old_timer_does_not_force_refresh_of_new_token():
    client = client_with_expired_token()
    client._config['oauth']['expires_in'] = (get_datetime() + timedelta(hours=1)).isoformat()
    client._oauth_token_scheduled = Mock()
    with patch.object(stellantis, 'async_track_point_in_time'):
        await client.scheduled_oauth_token_refresh(get_datetime())
    client.refresh_oauth_token_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_in_flight_never_rearms_timer():
    client = client_with_expired_token()
    async def refresh():
        client._shutting_down = True
    client.refresh_oauth_token_request.side_effect = refresh
    with patch.object(stellantis, 'async_track_point_in_time') as timer:
        await client.scheduled_oauth_token_refresh(get_datetime())
        timer.assert_not_called()


@pytest.mark.asyncio
async def test_rejected_refresh_token_is_not_sent_again():
    client = stellantis.StellantisVehicles(Mock())
    client.save_config({'oauth': {'refresh_token': 'synthetic'}})
    client.make_http_request = AsyncMock(side_effect=ConfigEntryAuthFailed('invalid_grant'))
    client.apply_query_params = Mock(return_value='https://example.invalid/token')
    client.apply_dict_params = Mock(return_value={})
    for _ in range(2):
        with pytest.raises(ConfigEntryAuthFailed):
            await client.refresh_oauth_token_request()
    client.make_http_request.assert_awaited_once()


def account():
    return {'mobile_app': 'MyPeugeot', 'country_code': 'DE', 'customer_id': 'fake-customer',
            FIELD_REMOTE_COMMANDS: True, FIELD_NOTIFICATIONS: False, FIELD_ANONYMIZE_LOGS: True,
            FIELD_OAUTH_CODE_URL: 'http://local-worker:3000',
            'oauth': {'access_token': 'old'}, 'mqtt': {'access_token': 'old-mqtt'},
            'vehicles': {'fake-vehicle': {'setting': True}}}


@pytest.mark.asyncio
@pytest.mark.parametrize('loaded', [False, True])
@pytest.mark.parametrize('action,step', [('oauth','oauth_mode'),('options','options'),
                                      (FIELD_REMOTE_COMMANDS,'otp')])
async def test_reconfigure_uses_own_client_even_when_entry_is_not_loaded(loaded, action, step):
    entry = SimpleNamespace(data=account())
    runtime = Mock()
    if loaded:
        entry.runtime_data = runtime
    flow = config_flow.StellantisVehiclesConfigFlow()
    flow.hass = Mock()
    flow.init_translations = AsyncMock()
    flow._get_reconfigure_entry = Mock(return_value=entry)
    setattr(flow, f'async_step_{step}', AsyncMock(return_value={'step_id':step}))
    result = await flow.async_step_reconfigure({FIELD_RECONFIGURE: action})
    assert result['step_id'] == step
    assert flow.stellantis is not runtime
    assert flow.stellantis.get_config('oauth') == entry.data['oauth']
    runtime.disable_remote_commands.assert_not_called()
    assert not {'oauth','mqtt','vehicles'} & flow.data.keys()
    assert flow.data[FIELD_OAUTH_CODE_URL] == 'http://local-worker:3000'


@pytest.mark.asyncio
async def test_reauth_preserves_preferences_without_overwriting_live_credentials():
    flow = config_flow.StellantisVehiclesConfigFlow()
    flow.async_step_reauth_confirm = AsyncMock()
    await flow.async_step_reauth(account())
    assert flow.data[FIELD_NOTIFICATIONS] is False
    assert flow.data[FIELD_REMOTE_COMMANDS] is True
    assert flow.data[FIELD_OAUTH_CODE_URL] == 'http://local-worker:3000'
    assert not {'oauth','mqtt','vehicles'} & flow.data.keys()


@pytest.mark.asyncio
async def test_failed_local_worker_keeps_url_and_allows_retry_without_password_storage():
    flow = config_flow.StellantisVehiclesConfigFlow()
    flow.stellantis = Mock()
    flow.stellantis.get_oauth_code = AsyncMock(side_effect=CommunicationError('offline'))
    flow.get_error_message = Mock(return_value='get_oauth_code')
    flow.async_show_form = Mock(side_effect=lambda **kwargs: kwargs)
    result = await flow.async_step_oauth_remote({
        'email': 'synthetic@example.invalid', 'password': 'synthetic-password',
        FIELD_OAUTH_CODE_URL: 'http://local-worker:3000',
    })
    assert result['step_id'] == 'oauth_remote'
    assert result['errors'] == {'base': 'get_oauth_code'}
    assert flow.data == {FIELD_OAUTH_CODE_URL: 'http://local-worker:3000'}


@pytest.mark.asyncio
async def test_cancelled_flow_closes_owned_session():
    flow = config_flow.StellantisVehiclesConfigFlow()
    tasks = []
    flow.hass = Mock()
    flow.hass.async_create_task.side_effect = lambda coro: tasks.append(asyncio.create_task(coro))
    client = Mock(close_session=AsyncMock())
    flow.stellantis = client
    flow.async_remove()
    await asyncio.gather(*tasks)
    client.close_session.assert_awaited_once()
    assert flow.stellantis is None
