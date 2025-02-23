from __future__ import annotations

import os
from typing import Any, Union, Mapping
from typing_extensions import Self, override

import httpx

from openai import _exceptions
from openai._qs import Querystring
from openai._types import (
    NOT_GIVEN,
    Omit,
    Timeout,
    NotGiven,
)
from openai._utils import (
    is_given,
    is_mapping,
)
from openai._version import __version__
from openai.resources import files, images, models, batches, embeddings, completions, moderations
from openai._streaming import Stream as Stream, AsyncStream as AsyncStream
from openai._exceptions import OpenAIError, APIStatusError
from openai._base_client import (
    DEFAULT_MAX_RETRIES,
    SyncAPIClient,
)
from openai.resources.beta import beta
from openai.resources.chat import chat
from openai.resources.audio import audio
from openai.resources.uploads import uploads
from openai.resources.fine_tuning import fine_tuning

class LocalAPIClient(SyncAPIClient):
    completions: completions.Completions
    chat: chat.Chat
    embeddings: embeddings.Embeddings
    files: files.Files
    images: images.Images
    audio: audio.Audio
    moderations: moderations.Moderations
    models: models.Models
    fine_tuning: fine_tuning.FineTuning
    beta: beta.Beta
    batches: batches.Batches
    uploads: uploads.Uploads
    with_raw_response: OpenAIWithRawResponse
    with_streaming_response: OpenAIWithStreamedResponse

    # client options
    api_key: str
    organization: str | None
    project: str | None

    websocket_base_url: str | httpx.URL | None
    """Base URL for WebSocket connections.

    If not specified, the default base URL will be used, with 'wss://' replacing the
    'http://' or 'https://' scheme. For example: 'http://example.com' becomes
    'wss://example.com'
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        websocket_base_url: str | httpx.URL | None = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        # Configure a custom httpx client.
        # We provide a `DefaultHttpxClient` class that you can pass to retain the default values we use for `limits`, `timeout` & `follow_redirects`.
        # See the [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
        http_client: httpx.Client | None = None,
        # Enable or disable schema validation for data returned by the API.
        # When enabled an error APIResponseValidationError is raised
        # if the API responds with invalid data for the expected schema.
        #
        # This parameter may be removed or changed in the future.
        # If you rely on this feature, please open a GitHub issue
        # outlining your use-case to help us decide if it should be
        # part of our public interface in the future.
        _strict_response_validation: bool = False,
    ) -> None:
        """Construct a new synchronous openai client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        """
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise OpenAIError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )
        self.api_key = api_key

        if organization is None:
            organization = os.environ.get("OPENAI_ORG_ID")
        self.organization = organization

        if project is None:
            project = os.environ.get("OPENAI_PROJECT_ID")
        self.project = project

        self.websocket_base_url = websocket_base_url

        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url is None:
            base_url = f"https://api.openai.com/v1"

        super().__init__(
            version=__version__,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            custom_headers=default_headers,
            custom_query=default_query,
            _strict_response_validation=_strict_response_validation,
        )

        self._default_stream_cls = Stream

        self.completions = completions.Completions(self)
        self.chat = chat.Chat(self)
        self.embeddings = embeddings.Embeddings(self)
        self.files = files.Files(self)
        self.images = images.Images(self)
        self.audio = audio.Audio(self)
        self.moderations = moderations.Moderations(self)
        self.models = models.Models(self)
        self.fine_tuning = fine_tuning.FineTuning(self)
        self.beta = beta.Beta(self)
        self.batches = batches.Batches(self)
        self.uploads = uploads.Uploads(self)
        self.with_raw_response = OpenAIWithRawResponse(self)
        self.with_streaming_response = OpenAIWithStreamedResponse(self)

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="brackets")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        api_key = self.api_key
        return {"Authorization": f"Bearer {api_key}"}

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            "X-Stainless-Async": "false",
            "OpenAI-Organization": self.organization if self.organization is not None else Omit(),
            "OpenAI-Project": self.project if self.project is not None else Omit(),
            **self._custom_headers,
        }

    def copy(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        websocket_base_url: str | httpx.URL | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
        http_client: httpx.Client | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._client
        return self.__class__(
            api_key=api_key or self.api_key,
            organization=organization or self.organization,
            project=project or self.project,
            websocket_base_url=websocket_base_url or self.websocket_base_url,
            base_url=base_url or self.base_url,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            http_client=http_client,
            max_retries=max_retries if is_given(max_retries) else self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    @override
    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        data = body.get("error", body) if is_mapping(body) else body
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=data)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(err_msg, response=response, body=data)

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(err_msg, response=response, body=data)

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=data)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=data)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(err_msg, response=response, body=data)

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=data)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(err_msg, response=response, body=data)
        return APIStatusError(err_msg, response=response, body=data)


class OpenAIWithRawResponse:
    def __init__(self, client: LocalAPIClient) -> None:
        self.completions = completions.CompletionsWithRawResponse(client.completions)
        self.chat = chat.ChatWithRawResponse(client.chat)
        self.embeddings = embeddings.EmbeddingsWithRawResponse(client.embeddings)
        self.files = files.FilesWithRawResponse(client.files)
        self.images = images.ImagesWithRawResponse(client.images)
        self.audio = audio.AudioWithRawResponse(client.audio)
        self.moderations = moderations.ModerationsWithRawResponse(client.moderations)
        self.models = models.ModelsWithRawResponse(client.models)
        self.fine_tuning = fine_tuning.FineTuningWithRawResponse(client.fine_tuning)
        self.beta = beta.BetaWithRawResponse(client.beta)
        self.batches = batches.BatchesWithRawResponse(client.batches)
        self.uploads = uploads.UploadsWithRawResponse(client.uploads)


class OpenAIWithStreamedResponse:
    def __init__(self, client: LocalAPIClient) -> None:
        self.completions = completions.CompletionsWithStreamingResponse(client.completions)
        self.chat = chat.ChatWithStreamingResponse(client.chat)
        self.embeddings = embeddings.EmbeddingsWithStreamingResponse(client.embeddings)
        self.files = files.FilesWithStreamingResponse(client.files)
        self.images = images.ImagesWithStreamingResponse(client.images)
        self.audio = audio.AudioWithStreamingResponse(client.audio)
        self.moderations = moderations.ModerationsWithStreamingResponse(client.moderations)
        self.models = models.ModelsWithStreamingResponse(client.models)
        self.fine_tuning = fine_tuning.FineTuningWithStreamingResponse(client.fine_tuning)
        self.beta = beta.BetaWithStreamingResponse(client.beta)
        self.batches = batches.BatchesWithStreamingResponse(client.batches)
        self.uploads = uploads.UploadsWithStreamingResponse(client.uploads)

