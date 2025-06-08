import asyncio
from dataclasses import dataclass, field
from agents import Runner, RunHooks
import threading
from typing import Any, Dict, Optional, List
import traceback
import time
from datetime import datetime
import numpy as np
import gradio as gr
from .config import logger
@dataclass
class RunnerStep:
    """Log entry for a single Runner step"""
    timestamp: str
    step_type: str
    agent_name: str
    turn_number: int
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None

    def __str__(self) -> str:
        return f"[{self.timestamp}][T{self.turn_number}][{self.step_type}]: {self.details}"

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
        if isinstance(self.data, np.ndarray):
            return gr.Image(self.data)
        return self.data


class RunnerLoggerHooks(RunHooks):
    """Custom hooks to log every step of the Runner"""
    
    def __init__(self, memory_instance):
        super().__init__()
        self.memory = memory_instance
        self.current_turn = 0
        self.turn_start_time = None
    
    async def on_agent_start(self, context, agent):
        self.current_turn += 1
        self.turn_start_time = time.time()
        
        step = RunnerStep(
            timestamp=datetime.now().isoformat(),
            step_type="turn_start",
            agent_name=agent.name,
            turn_number=self.current_turn,
            details={"message": f"Starting turn {self.current_turn} with agent {agent.name}"}
        )
        self.memory.log_runner_step(step)
    
    async def on_agent_end(self, context, agent, result):
        if self.turn_start_time:
            duration = (time.time() - self.turn_start_time) * 1000
        else:
            duration = None
            
        step = RunnerStep(
            timestamp=datetime.now().isoformat(),
            step_type="agent_call",
            agent_name=agent.name,
            turn_number=self.current_turn,
            details={"message": f"Agent {agent.name} completed", "result_type": type(result).__name__},
            duration_ms=duration
        )
        self.memory.log_runner_step(step)
    
    async def on_tool_start(self, context, agent, tool_call):
        tool_name = getattr(tool_call, 'name', 'unknown')
        tool_args = None
        for attr in ['arguments', 'args', 'function', 'parameters']:
            if hasattr(tool_call, attr):
                tool_args = getattr(tool_call, attr)
                break
        step = RunnerStep(
            timestamp=datetime.now().isoformat(),
            step_type="tool_call",
            agent_name=agent.name,
            turn_number=self.current_turn,
            details={
                "tool_name": tool_name,
                "tool_args": tool_args,
                "message": f"Calling tool {tool_name}"
            }
        )
        self.memory.log_runner_step(step)
    
    async def on_tool_end(self, context, agent, tool_call, result):
        # Handle different tool_call object attributes safely
        tool_name = getattr(tool_call, 'name', 'unknown')
        
        step = RunnerStep(
            timestamp=datetime.now().isoformat(),
            step_type="tool_result",
            agent_name=agent.name,
            turn_number=self.current_turn,
            details={
                "tool_name": tool_name,
                "result_length": len(str(result)) if result else 0,
                "message": f"Tool {tool_name} completed"
            }
        )
        self.memory.log_runner_step(step)


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
        self.limit: int = limit
        self.frames: list[Any] = []  
        self.snapshots: list[Any] = []      
        self.inputs: list[Any] = [] 
        self.chat = Chat()

        self.runner_steps: List[RunnerStep] = []
        self.step_limit: int = 1000  # Keep last 1000 steps
        self.logger_hooks: Optional[RunnerLoggerHooks] = None

        self._chat_q: asyncio.Queue[Any] = asyncio.Queue()
        self._input_q: asyncio.Queue[Any] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.agent: Any | None = None
        self.is_waiting: bool = False
        self.is_running: bool = False

    def log_runner_step(self, step: RunnerStep) -> None:
        """Log a runner step and maintain the step history limit"""
        self.runner_steps.append(step)
        logger.debug(f"[ 🛠️ ]{step}")
        while len(self.runner_steps) > self.step_limit:
            self.runner_steps.pop(0)

    def enqueue(self, data: Any) -> None:
        self.frames.append(data)
        while len(self.frames) > self.limit:
            self.frames.pop(0)
        return self.snapshots.pop(0) if self.snapshots else None
    
    def receive(self, text: str) -> None:
        self.chat.append(Message.user(text))
        self._loop.call_soon_threadsafe(self._chat_q.put_nowait, text)


    def setup(self, agents) -> None:
        """Bind *agent* and spawn the background monitor threads."""
        self.v_agent = agents
        self.logger_hooks = RunnerLoggerHooks(self)
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
            logger.debug(f"Processing: {text}")
            start_step = RunnerStep(
                timestamp=datetime.now().isoformat(),
                step_type="processing_start",
                agent_name=getattr(self.v_agent, 'name', 'unknown'),
                turn_number=0,
                details={"user_input": text}
            )
            self.log_runner_step(start_step)
            
            try:
                self.is_running = True
                result = await Runner.run(
                    starting_agent=self.v_agent,
                    input=text,
                    context=self,
                    hooks=self.logger_hooks  # Add our custom hooks here
                )
                
                self.is_running = False
                
                # Log successful completion
                success_step = RunnerStep(
                    timestamp=datetime.now().isoformat(),
                    step_type="final_output",
                    agent_name=getattr(self.v_agent, 'name', 'unknown'),
                    turn_number=self.logger_hooks.current_turn if self.logger_hooks else 0,
                    details={
                        "output_type": type(result.final_output).__name__,
                        "output_preview": str(result.final_output)[:100] + "..." if len(str(result.final_output)) > 100 else str(result.final_output)
                    }
                )
                self.log_runner_step(success_step)
                
            except Exception as exc:  # noqa: BLE001
                self.is_running = False
                full_traceback = traceback.format_exc()
                logger.debug(f"Error in _monitor_chat: {exc}\n{full_traceback}")
                
                # Log the error
                error_step = RunnerStep(
                    timestamp=datetime.now().isoformat(),
                    step_type="error",
                    agent_name=getattr(self.v_agent, 'name', 'unknown'),
                    turn_number=self.logger_hooks.current_turn if self.logger_hooks else 0,
                    details={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": full_traceback
                    }
                )
                self.log_runner_step(error_step)
                continue
            final = result.final_output.split('</think>', 1)[-1]
            self.chat.append(Message.assistant(final))
            await asyncio.sleep(0)
