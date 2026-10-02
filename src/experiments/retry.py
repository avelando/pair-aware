DEFAULT_MAX_RETRIES = 0


def validate_max_retries(
    max_retries,
):
    if (
        isinstance(
            max_retries,
            bool,
        )
        or not isinstance(
            max_retries,
            int,
        )
        or max_retries < 0
    ):
        raise ValueError(
            "max_retries must be a non-negative integer."
        )

    return max_retries