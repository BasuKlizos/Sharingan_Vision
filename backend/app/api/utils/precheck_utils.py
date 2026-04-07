from app.modules.precheck.service import LightingPrecheckService


def get_lighting_precheck_service() -> LightingPrecheckService:
    """Provide lighting precheck service instance."""
    return LightingPrecheckService()
