from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Literal

Player = int
Owner = Optional[int]
NodeId = int


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float


@dataclass(frozen=True)
class GameNode:
    id: NodeId
    pos: Vec2
    owner: Owner
    strength: float


@dataclass(frozen=True)
class Edge:
    a: NodeId
    b: NodeId


@dataclass(frozen=True)
class Flow:
    src: NodeId
    dst: NodeId
    player: Player


@dataclass(frozen=True)
class GameState:
    nodes: tuple[GameNode, ...]
    edges: tuple[Edge, ...]
    flows: tuple[Flow, ...]
    tick: int
    num_players: int


@dataclass(frozen=True)
class Action:
    kind: Literal["toggleFlow"]
    src: NodeId
    dst: NodeId
    player: Player


REGEN_PER_SEC = 1.1
TRANSFER_PER_SEC = 3.0
MAX_STRENGTH = 100.0
MIN_STRENGTH_TO_SEND = 0.1
ATTACK_BONUS = 0.5
