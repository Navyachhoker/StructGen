from invoice_extractor.clients.router import ModelRouter


class SuccessfulClient:
    def extract_raw(self, prompt: str) -> str:
        return '{"source": "primary"}'


class FailingClient:
    def extract_raw(self, prompt: str) -> str:
        raise RuntimeError("Primary model failed")


class FallbackClient:
    def extract_raw(self, prompt: str) -> str:
        return '{"source": "fallback"}'


def test_router_uses_primary_when_successful():
    router = ModelRouter(
        primary=SuccessfulClient(),
        fallback=FallbackClient(),
        primary_name="qwen-invoice-lora",
        fallback_name="openai/gpt-oss-120b",
    )

    result = router.extract_raw("test prompt")
    metadata = router.get_last_metadata()

    assert result == '{"source": "primary"}'
    assert metadata is not None
    assert metadata.model_used == "qwen-invoice-lora"
    assert metadata.fallback_used is False


def test_router_uses_fallback_when_primary_fails():
    router = ModelRouter(
        primary=FailingClient(),
        fallback=FallbackClient(),
        primary_name="qwen-invoice-lora",
        fallback_name="openai/gpt-oss-120b",
    )

    result = router.extract_raw("test prompt")
    metadata = router.get_last_metadata()

    assert result == '{"source": "fallback"}'
    assert metadata is not None
    assert metadata.model_used == "openai/gpt-oss-120b"
    assert metadata.fallback_used is True