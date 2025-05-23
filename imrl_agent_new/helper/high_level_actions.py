from enum import IntEnum


class HighLevelActions(IntEnum):
    GO_ONION, GO_TOMATO, GO_DISH, PUT_ONION, PUT_TOMATO, GO_READY_POT, GO_SERVE, GO_COUNTER, START_COOKING, WAIT = range(
        10)
