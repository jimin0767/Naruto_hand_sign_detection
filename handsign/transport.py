"""UDP sockets for the Unity bridge.

Deliberately thin. Everything worth testing lives in :mod:`handsign.bridge`; this module
only moves bytes, so it stays small enough to review at a glance.

Both sockets are non-blocking or timed out. The recognition loop must never stall waiting
on the network -- a frame missed while blocked on a socket is a frame the player's hand
was not seen.
"""

from __future__ import annotations

import socket
from typing import Iterable


class UdpUplink:
    """Fire-and-forget sender to Unity's JutsuReceiver."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5010):
        self.address = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)

    def send(self, payload: bytes) -> None:
        """Send one datagram, swallowing the error if nothing is listening yet.

        Unity may not be running, or may be between scenes. That is not a reason to stop
        recognising -- the player can start the game after the bridge.
        """
        try:
            self._socket.sendto(payload, self.address)
        except OSError:
            pass

    def send_all(self, payloads: Iterable[bytes]) -> None:
        for payload in payloads:
            self.send(payload)

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "UdpUplink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class UdpDownlink:
    """Non-blocking receiver for Unity's control messages."""

    def __init__(self, port: int = 5011, host: str = "0.0.0.0"):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, port))
        self._socket.setblocking(False)
        self.port = port

    def poll(self, max_messages: int = 16) -> list[bytes]:
        """Drain whatever has arrived since the last call.

        Bounded so a flood cannot starve the recognition loop.
        """
        out: list[bytes] = []
        for _ in range(max_messages):
            try:
                data, _addr = self._socket.recvfrom(2048)
            except BlockingIOError:
                break
            except OSError:
                break
            out.append(data)
        return out

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "UdpDownlink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
