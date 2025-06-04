from agents import Runner
import cv2
import gradio as gr
from fastrtc import Stream,WebRTC
from app.config import env
import gradio as gr
from fastrtc import AdditionalOutputs
from app.memory import Memory,Message
from fastrtc import Stream, get_cloudflare_turn_credentials
from app.agent import build_agent

# Initialize global memory instance
mem = Memory()

def video_handler(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if (snapshot := mem.enqueue(frame)):
        mem.chat.append(Message.tool(snapshot.gr, title=snapshot.sender, status='done'))
    return frame,AdditionalOutputs(mem.chat.messages)

def chat_handler(text, request: gr.Request):
    if not mem.is_running:
        mem.receive(text)
    return ""

def render(stream):
    video = WebRTC(
        label="Stream",
        rtc_configuration=get_cloudflare_turn_credentials(hf_token=env.hf_token),
        track_constraints={"width": {"exact": 500}, "height": {"exact": 500}, "aspectRatio": {"exact": 1}},
        mode="send",
        modality="video",
        mirror_webcam=True,
        width=500,
        height=500,
    )
    stream.webrtc_component = video
    video.stream(
        fn=stream.event_handler,
        inputs=[video],
        outputs=[video],
        time_limit=stream.time_limit,
    )

    video.on_additional_outputs(
        stream.additional_outputs_handler,
        inputs=stream.additional_output_components,
        outputs=stream.additional_output_components,
    )




if __name__ == "__main__":
    shared_chatbot = gr.Chatbot(type="messages", height=400, autoscroll=False)

    v_stream = Stream(
        video_handler,
        modality="video",
        mode="send",
        additional_outputs=[shared_chatbot],
        additional_outputs_handler=lambda _o1,n1: (n1),
    )

    with gr.Blocks() as demo:

        # Setup the global memory instance
        mem.setup(build_agent())
        gr.Markdown("## Perceptual CoPilot Demo")

        with gr.Row(): 
            with gr.Column(scale=1): 
                render(v_stream)
            with gr.Column(scale=1): 
                with gr.Group(): 
                    shared_chatbot.render()
                    chat_input = gr.Textbox(
                        placeholder="Type your message here and press Enter",
                        label="Chat Input",
                        lines=1,
                    )
                    chat_input.submit(
                        chat_handler,
                        inputs=[chat_input],
                        outputs=[chat_input],
                    )

        

    demo.launch(server_name="0.0.0.0", server_port=17788)
    # demo.launch(share=True)