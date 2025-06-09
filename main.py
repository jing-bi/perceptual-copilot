from pathlib import Path
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
        session_memories[session_id] = Memory(build_agent())    
    return session_memories[session_id]

def video_handler(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rtcid = get_current_context().webrtc_id
    mem = get_session_memory(rtcid)
    if (snapshot := mem.enqueue(frame)):
        mem.chat.append(Message.tool(snapshot.gr, title=snapshot.sender, status='done'))
    return frame, AdditionalOutputs(mem.chat.messages, rtcid)

def chat_handler(text, webrtc_state):
    # Handle case where webrtc_state might be None
    if webrtc_state is None:
        return "", [{"role": "assistant", "content": "Please start your camera first to begin the conversation."}], webrtc_state
    
    mem = get_session_memory(webrtc_state)
    if not mem.is_running:
        mem.receive(text.strip())
    return "", mem.chat.messages, webrtc_state





if __name__ == "__main__":
    

    with gr.Blocks(
        title="🤖 Perceptual Copilot - AI Vision Assistant", 
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="orange", 
            neutral_hue="slate",
            font=("system-ui", "sans-serif")
        ),
        css=Path("styles.css").read_text(),
    ) as demo:
        
        # Header section with enhanced styling
        gr.Markdown("""
        <div style="text-align: center;">
            <h1>🤖 Perceptual Copilot</h1>
            <p style="font-size: 1.2em; color: #666; margin: 10px 0;">
                <span class="status-indicator"></span>
                This is an experimental prototype that integrates OpenAI agents with visual tools to process real-time video streams.
            </p>
            <p style="color: #888;">
                Real-time video analysis • Visual agents • Interactive chat
            </p>
        </div>
        """)
        
        state = gr.State(value=None)
        
        # Main interface with improved layout
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="video-container"):
                video = WebRTC(
                    label="🎥 Camera Stream",
                    # rtc_configuration=get_cloudflare_turn_credentials(hf_token=env.hf_token),
                    track_constraints={"width": {"exact": 600}, "height": {"exact": 600}, "aspectRatio": {"exact": 1}},
                    mode="send",
                    modality="video",
                    mirror_webcam=True,
                    width=600,
                    height=600,
                )
            
            with gr.Column(scale=1, elem_classes="chat-container"):
                gr.Markdown("### 💬 AI Assistant Chat")
                chatbot = gr.Chatbot(
                    type="messages", 
                    height=450,
                    label="🤖 AI Assistant",
                    placeholder="Chat history will appear here...",
                    show_label=False,
                )
                
                with gr.Row():
                    textbox = gr.Textbox(
                        placeholder="💭 Question goes here",
                        label="Chat Input",
                        lines=1,
                        show_label=False
                    )
        # Event handlers
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
        
        # Chat handler for textbox
        textbox.submit(
            chat_handler,
            inputs=[textbox, state],
            outputs=[textbox, chatbot, state]
        )

        
        # Enhanced instructions section
        with gr.Column(elem_classes="instructions-container"):
            gr.Markdown("""
            ## 🚀 Get Started
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 💡 Example Prompts
                    
                    **🌍 General Vision:**
                    - *"What do you see in front of me?"*
                    - *"What's the overall environment like?"*
                    
                    **📄 Text & Documents:**
                    - *"Read the text in this document"*
                    - *"Extract the code snippet from this image"*
                    
                    **🔍 Object Recognition:**
                    - *"What objects are visible?"*
                    - *"Help me identify this item"*
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 🔧 Current Capabilities
                    
                    **🚀 Available Features:**
                    - **OCR** - Text extraction and reading
                    - **Q&A** - Visual question answering
                    - **Caption** - Scene description and analysis
                    - **Localization** - Object detection and positioning
                    - **Time** - Current time and temporal context
                    
                    **📈 More Coming Soon:**
                    We're continuously adding new capabilities to enhance your visual AI experience.
                    
                    **⚠️ Important Note:**
                    All models are self-hosted. Please avoid abuse of the system.
                    """)
    demo.queue(default_concurrency_limit=None)
    demo.launch(server_name="0.0.0.0", server_port=17788)