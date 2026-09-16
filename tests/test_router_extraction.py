from invoice_extractor.clients.fake import FakeModelClient
from invoice_extractor.clients.router import ModelRouter
from invoice_extractor.extraction import extract_invoice


class FailingModelClient(FakeModelClient):
    """Fake primary model that always fails."""

    def extract_raw(self, prompt: str) -> str:
        raise RuntimeError("Primary model unavailable")


def test_router_works_with_extraction_pipeline():
    primary = FakeModelClient()
    fallback = FakeModelClient()

    router = ModelRouter(
        primary=primary,
        fallback=fallback,
        primary_name="qwen-invoice-lora",
        fallback_name="openai/gpt-oss-120b",
    )

    result = extract_invoice(
        raw_text="Sample invoice from Fake Vendor Inc.",
        client=router,
    )

    metadata = router.get_last_metadata()

    assert result.success is True
    assert result.invoice is not None

    assert metadata is not None
    assert metadata.model_used == "qwen-invoice-lora"
    assert metadata.fallback_used is False


def test_router_falls_back_when_primary_fails():
    primary = FailingModelClient()
    fallback = FakeModelClient()

    router = ModelRouter(
        primary=primary,
        fallback=fallback,
        primary_name="qwen-invoice-lora",
        fallback_name="openai/gpt-oss-120b",
    )

    result = extract_invoice(
        raw_text="Sample invoice from Fake Vendor Inc.",
        client=router,
    )

    metadata = router.get_last_metadata()

    assert result.success is True
    assert result.invoice is not None

    assert metadata is not None
    assert metadata.model_used == "openai/gpt-oss-120b"
    assert metadata.fallback_used is True