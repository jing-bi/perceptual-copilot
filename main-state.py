import cv2
import gradio as gr
from fastrtc import Stream,WebRTC
from app.config import env
from fastrtc import AdditionalOutputs
from app.memory import Memory,Message
from fastrtc import get_cloudflare_turn_credentials
from app.agent import build_agent
from fastrtc import get_current_context
session_memories = {}

def get_session_memory(session_id: str = None) -> Memory:
    if session_id not in session_memories:
        memory = Memory()
        memory.setup(build_agent())
        session_memories[session_id] = memory
    return session_memories[session_id]

def video_handler(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_webrtc_id = get_current_context().webrtc_id
    print(f"Processing frame for WebRTC ID: {current_webrtc_id}")
    mem = get_session_memory(current_webrtc_id)
    
    if (snapshot := mem.enqueue(frame)):
        mem.chat.append(Message.tool(snapshot.gr, title=snapshot.sender, status='done'))
    return frame, AdditionalOutputs(mem.chat.messages, current_webrtc_id)

def chat_handler(text, webrtc_state, request: gr.Request):
    text = text.strip()
    print(f"Received text: {text}, WebRTC State: {webrtc_state}")
    mem = get_session_memory(webrtc_state)
    if not mem.is_running:
        mem.receive(text)
    return "", mem.chat.messages





if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## Perceptual CoPilot Demo")
        state = gr.State(value=None)
        video = WebRTC(
            label="Stream",
            # rtc_configuration=get_cloudflare_turn_credentials(hf_token=env.hf_token),
            track_constraints={"width": {"exact": 500}, "height": {"exact": 500}, "aspectRatio": {"exact": 1}},
            mode="send",
            modality="video",
            mirror_webcam=True,
            width=800,
            height=800,
        )
        chatbot = gr.Chatbot(type="messages", height=400)
        textbox = gr.Textbox(
            placeholder="Type your message here and press Enter",
            label="Chat Input",
            lines=1,
        )
        video.stream(
            fn=video_handler,
            inputs=[video],
            outputs=[video],
            concurrency_limit=10,
        )
        video.on_additional_outputs(
            fn=lambda messages, webrtc_id: (messages, webrtc_id),
            outputs=[chatbot, state]
        )
        textbox.submit(
            chat_handler,
            inputs=[textbox, state],
            outputs=[textbox, chatbot],
        )

    demo.launch(server_name="0.0.0.0", server_port=17788)