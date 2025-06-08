---
title: Perceptual Copilot
emoji: 👁️
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: main.py
pinned: false
license: mit
---

## ✨ What is Perceptual Copilot?

Perceptual Copilot is a prototype that demonstrates the integration of OpenAI agents with visual tools to process real-time video streams. This experimental platform showcases both the promising potential and current limitations of equipping agents with vision capabilities to understand and interact with live visual data. 


### Architecture Overview



```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      Webcam     │───▶│      Memory     │◀──▶│      Gradio     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │      Agent      │◀──▶│      Tools      │
                       └─────────────────┘    └─────────────────┘
```

### Available Tools

| Tool | Description | Output |
|------|-------------|---------|
| `caption` | Generate detailed image descriptions | Rich visual descriptions |
| `ocr` | Extract text from images | Extracted text content |
| `localize` | Detect and locate objects | Bounding boxes with labels |
| `qa` | Answer questions about images | Contextual answers |
| `time` | Get current timestamp | Current date and time |
| _More tools coming soon..._ | Additional capabilities in development | Various outputs |

## 🚀 Quick Start

### Prerequisites

- Webcam access

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   ```bash
   export HF_TOKEN="your_huggingface_token"
   export API_KEY="your_openai_api_key"
   export END_LANG="your_llm_endpoint"
   export END_TASK="your_task_endpoint"
   export MODEL_AGENT="your_agent_model"
   export MODEL_MLLM="your_multimodal_model"
   export MODEL_LOC="your_localization_model"
   ```

3. **Launch the application**
   ```bash
   python main.py
   ```

## 💡 Usage Examples

### Basic Interaction
- **User**: "What do you see?"
- **Assistant**: *Generates detailed caption of current view*

### OCR Functionality
- **User**: "Read the text in this document"
- **Assistant**: *Extracts and returns all visible text*

### Object Detection
- **User**: "What objects are in front of me?"
- **Assistant**: *Identifies and localizes objects with bounding boxes*


## Acknowledgments

- Built with [Gradio](https://gradio.app/) for the interactive web interface
- Uses [Supervision](https://supervision.roboflow.com/) for frame annotation
- WebRTC integration via [FastRTC](https://github.com/gradio-app/gradio)

