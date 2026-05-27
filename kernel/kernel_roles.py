"""
TOVAH v14 kernel/kernel_roles.py — Shared enums for the kernel ecology.

The sovereign kernel remains authoritative. Hub and subkernel layers are
explicitly typed so orchestration does not drift into ad-hoc strings.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Type, TypeVar


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class KernelRole(_StrEnum):
    MAIN = "main"
    HUB = "hub"
    SUBKERNEL = "subkernel"


class KernelLifecycle(_StrEnum):
    BORN = "born"
    MIRRORING = "mirroring"
    EXPERIMENTAL = "experimental"
    DEGRADED = "degraded"
    REVERTING = "reverting"
    STABLE_BRANCH = "stable_branch"
    PROMOTABLE = "promotable"
    RETIRED = "retired"


class TrustLevel(_StrEnum):
    UNTRUSTED = "untrusted"
    LOW = "low"
    PROVISIONAL = "provisional"
    TRUSTED = "trusted"
    SOVEREIGN = "sovereign"


class RiskClass(_StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(_StrEnum):
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class BootMode(_StrEnum):
    MAIN_ONLY = "main_only"
    MAIN_WITH_HUB = "main_with_hub"
    DISTRIBUTED_READY = "distributed_ready"


E = TypeVar("E", bound=Enum)


def enum_values(enum_cls: Type[E]) -> List[str]:
    return [str(v.value) for v in enum_cls]


def normalize_enum_value(enum_cls: Type[E], value: str | Enum) -> E:
    if isinstance(value, enum_cls):
        return value
    for item in enum_cls:
        if item.value == value:
            return item
    raise ValueError(f"invalid {enum_cls.__name__}: {value}")


def normalize_many(enum_cls: Type[E], values: Iterable[str | Enum]) -> List[E]:
    return [normalize_enum_value(enum_cls, value) for value in values]
