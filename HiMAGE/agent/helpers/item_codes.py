from enum import IntEnum


class ItemCode(IntEnum):
    """Compact integer codes for objects the chef can hold or deliver."""
    NOTHING = 0
    ONION = 1
    TOMATO = 2
    DISH = 3
    SOUP = 4

    @classmethod
    def ItemCodeName(cls, value: int) -> str:
        """
        Get the name of the item code by its enum value.
        :param value: The enum value of the item code.
        :return: The name of the item code.
        """
        return cls(value).name if value in cls._value2member_map_ else "UNKNOWN"

    @classmethod
    def ItemCodeValue(cls, name: str) -> int:
        """
        Get the enum value of the item code by its name.
        :param name: The name of the item code.
        :return: The enum value of the item code, or -1 if not found.
        """
        # Allow for case-insensitive lookup
        name = name.upper()
        return cls[name].value if name in cls._member_map_ else -1
