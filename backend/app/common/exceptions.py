class SignalingError(Exception):
    pass


class RoomFullError(SignalingError):
    pass


class InvalidMessageError(SignalingError):
    pass