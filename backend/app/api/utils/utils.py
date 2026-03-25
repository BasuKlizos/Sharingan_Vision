import uuid

def generate_session_id() -> str:
    """
    Generate a unique session identifier.

    Returns:
        str: UUID4 string.
    """
    return str(uuid.uuid4())