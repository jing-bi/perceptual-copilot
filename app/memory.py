import asyncio
from dataclasses import dataclass, field
from agents import Runner
import threading
import threading
from typing import Any, Dict, Optional
import traceback
@dataclass
class Message:
    role: str
    content: str
    mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content, '')
    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content, '')

    @classmethod
    def tool(cls, content: str, **kwargs) -> "Message":
        return cls("assistant", content, 'tool', kwargs)
    @classmethod
    def assistant(cls, content: str, mode='') -> "Message":
        return cls("assistant", content, mode)
    @classmethod
    def tts(cls, content: str) -> "Message":
        return cls("assistant", content, 'tts')
    def to_dict(self) -> Dict[str, Any]:

        result = {"role": self.role, "content": self.content}
        if self.mode == "tool":
            metadata = self.metadata.copy()
            if title := metadata.get("title"):
                metadata["title"] = title.title()
            result["metadata"] = metadata
        return result


@dataclass
class Snapshot:
    sender: str
    data: Any
    @property
    def gr(self):
        return gr.Image(self.data) if not isinstance( self.data, str) else self.data

class Chat:
    def __init__(self):
        self.history = [] 

    def append(self, message: Message):
        self.history.append(message)
    @property
    def messages(self):
        return [i.to_dict() for i in self.history]

    


class Memory:


    def __init__(self, limit: int = 200) -> None:
        # ---- data windows -------------------------------------------------
        self.limit: int = limit
        self.frames: list[Any] = []          # rolling interaction buffer
        self.snapshots: list[Any] = []       # finished schedule results
        self.inputs: list[Any] = [] 
        self.chat = Chat()

        self._chat_q: asyncio.Queue[Any] = asyncio.Queue()
        self._input_q: asyncio.Queue[Any] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.agent: Any | None = None
        self.is_waiting: bool = False
        self.is_running: bool = False


    def enqueue(self, data: Any) -> None:
        self.frames.append(data)
        while len(self.frames) > self.limit:
            self.frames.pop(0)
        return self.snapshots.pop(0) if self.snapshots else None
    

    def receive(self, text: str) -> None:
        print(f"Received user input: {text}")
        self.chat.append(Message.user(text))
        self._loop.call_soon_threadsafe(self._chat_q.put_nowait, text)

    def schedule(self, item: Any) -> None:
        self._run_history.append(f"[📝][Added] {item}")
        self._loop.call_soon_threadsafe(self._schedule_q.put_nowait, item)

    def setup(self, agents) -> None:
        """Bind *agent* and spawn the background monitor threads."""
        self.v_agent = agents

        def _runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.create_task(self._monitor_chat())
                self._loop.run_forever()
            finally:
                self._loop.close()

        threading.Thread(target=_runner, daemon=True).start()
    async def _monitor_chat(self) -> None:
        """Process incoming chat messages, respecting the waiting gate."""
        while True:
            text = await self._chat_q.get()
            print(f"Processing user input: {text}")
            try:
                self.is_running = True
                result = await Runner.run(
                    starting_agent=self.v_agent,
                    input=text,
                    context=self
                )
                self.is_running = False
            except Exception as exc:  # noqa: BLE001
                full_traceback = traceback.format_exc()
                print(f"Error in _monitor_chat: {exc}\n{full_traceback}")
                continue

            final = result.final_output.split('</think>', 1)[-1]
            self.chat.append(Message.assistant(final))
            await asyncio.sleep(0)


