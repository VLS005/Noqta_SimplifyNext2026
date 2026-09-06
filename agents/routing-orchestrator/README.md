# SimplifyNext 2026: Agentic Routing System

This repository contains a cutting-edge multi-agent routing system designed to provide safe, accessible, and highly-contextualized navigation for visually impaired users.

## How to Run the System

The backend is built with FastAPI. To start the `RoutingAgent` and the entire API server, run the following command from the root directory:

```bash
uvicorn backend.main:app --reload --port 8000
```

Once running, you can interact with the agentic endpoints (like triggering navigation or the vision assistant) via the Swagger UI at `http://127.0.0.1:8000/docs` or via `curl`.

---

## The Multi-Agent Architecture

This system relies on a swarm of specialized, autonomous sub-agents coordinated by a central orchestrator.

### 1. `RoutingAgent` (The Orchestrator)
The central intelligence of the backend. It receives the user's request and orchestrates a complex pipeline of sub-agents to find, score, translate, and execute the best possible route. It also manages the **Live Navigation Simulator**, looping through instructions and pausing them if safety alerts arise.

### 2. `DirectionFinderAgent`
Responsible for the raw geographical data. It calls external providers (like Google Routes API and OpenStreetMap) to generate multiple candidate paths between the origin and destination.

### 3. `AccessibilityAgent`
The safety evaluator. It ranks and scores the candidate routes based on the user's specific mobility profile, penalizing routes with high crowding, poor lighting, or lack of tactile paving, and rewarding step-free access.

### 4. `ETAAgent`
Calculates highly personalized Estimated Times of Arrival (ETA). Instead of relying on generic walking speeds, it queries DynamoDB to look up the user's historical walking pace and adjusts the ETA accordingly.

### 5. `InstructionParsingAgent`
The linguistic translator. It takes generic GPS directions (e.g., "Turn left in 50m") and uses Amazon Bedrock (Nova Lite) to rewrite them into detailed, blind-friendly micro-instructions. It also handles advanced multimodal analysis (Nova Pro) to turn raw vision data into actionable safety warnings.

### 6. `VisionAgent`
The real-time spatial awareness system. It uses OpenCV to capture the user's live webcam feed. 
- **Layer 1:** It uses AWS Rekognition to quickly detect objects (with an API throttle of 1 call every 3 seconds to protect budget).
- **Layer 2:** If hazards are detected, it forwards the scene to Amazon Nova Pro for deep spatial reasoning before passing it to the parsing agent.

### 7. `VoiceAgent`
The vocal interface. It utilizes Amazon Polly to generate Text-To-Speech (TTS) audio on the fly. It is responsible for speaking the navigation instructions aloud and interrupting them with sharp haptic pings and voice warnings if the VisionAgent detects an obstacle.
