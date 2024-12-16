class NetworkValidatorException(Exception):
    pass


class NetworkValidatorExceptionGroup(NetworkValidatorException, ExceptionGroup):
    pass
