# AndroidWorld Agents

## Overview

The agents in this folder are designed to interact with Android devices by
perceiving the device state and taking appropriate actions to accomplish user
goals. The framework uses a hierarchical structure with a base agent class that
specialized agents extend.

## Agents

### M3A (Multimodal Autonomous Agent for Android)

- Featured in the AndroidWorld paper (ICLR 2025)
- Uses both visual and textual data to interact with Android devices
- Capable of understanding screenshots and text descriptions of UI elements
- Uses Set-of-Mark action space

### T3A (Text-only Autonomous Agent for Android)

- Featured in the AndroidWorld paper (ICLR 2025)
- Text-only version of M3A
- Uses only textual representations of UI elements without visual information
- Works with the same action space as M3A

### SeeAct

- Featured in the AndroidWorld paper (ICLR 2025)
- Web-adapted version of the agent from "GPT-4V(ision) is a Generalist Web Agent, if Grounded"
- Uses visual grounding with a two-step reasoning process
- Specialized for interacting with Android interfaces

### Base and Utility Classes

- `base_agent.py`: Abstract base class defining the agent interface
- `agent_utils.py`: Common utility functions used by agents
- `infer.py`: Inference utilities for working with language models
- `*_utils.py`: Agent-specific utility functions

## Usage

Agents implement the `EnvironmentInteractingAgent` interface, providing a
consistent way to:

- Get the device state
- Take actions on the device
- Process feedback after actions
- Track progress through a series of interactions

Each agent has specific initialization requirements but follows the same basic
interaction pattern through the `step()` method.