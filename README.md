# Hierarchical VQA Pedestrian Navigation System

A Visual Question Answering (VQA) system that uses hierarchical questioning to analyze walking routes and detect pedestrian hazards efficiently.

## Overview

This system processes image sequences and asks safety-critical questions about the pedestrian environment. It uses a two-level hierarchical approach:
- **Level 1**: 8 primary questions to detect general conditions (always asked)
- **Level 2**: Detailed follow-up questions (only asked if Level 1 answer is "Yes")

This reduces redundant questions from 71 to typically 10-30 per frame.

## Quick Start

### Installation

```bash
pip install torch torchvision transformers Pillow opencv-python
pip install salesforce-lavis  # For BLIP2/InstructBLIP
```

### Basic Usage

```python
python app/runner.py
```

Edit configuration in `app/runner.py`:

```python
MODEL = "instructblip"              # vilt, blip2, llava, instructblip
PROMPT_PRESET = "full_hierarchical" # See presets below
CITY = "Singapore"
```

## Hierarchical Structure

### Level 1 Questions (Always Asked)
1. Is there a crossing nearby?
2. Are stairs visible?
3. Is construction visible?
4. Is any obstacle blocking the path?
5. Are multiple pedestrians present?
6. Is any vehicle nearby?
7. Is the surface hazardous?
8. Is visibility reduced?

### Level 2 Questions (Conditional)
Each "Yes" answer triggers 3-8 follow-up questions:
- **Crossing** → Signal state, traffic, markings
- **Stairs** → Direction, handrail, alternatives
- **Construction** → Barriers, equipment, detours
- **Obstacle** → Type, severity, navigation
- **Crowding** → Density, flow, impact
- **Vehicle** → Type (bike/car/motorcycle), movement, location
- **Surface** → Wetness, slope, damage
- **Visibility** → Time, glare, shadows

<img width="1612" height="1102" alt="VQA_jerarquia drawio(1)(1)" src="https://github.com/user-attachments/assets/bf77248f-4d39-48bc-8b37-04d94595bc05" />

## Configuration

### Available Presets

```python
PROMPT_PRESET = "level_1_only"        # Only 8 primary questions
PROMPT_PRESET = "crossing"            # Level 1 + crossing details
PROMPT_PRESET = "obstacle"            # Level 1 + obstacle details
PROMPT_PRESET = "vehicle"             # Level 1 + vehicle details
PROMPT_PRESET = "full_hierarchical"   # All questions (adaptive)
```

## Output

### Answer Log (answers.jsonl)

```json
{
  "question_id": "q_crossing_nearby",
  "answer": "yes",
  "confidence": 0.92,
  "level": 1,
  "parent_question": null
}
```

### Interactive Map

When GPS data is available, generates `interactive_map.html` with:
- Route visualization
- Event markers (color-coded)
- Image previews

## Example Output

```
Level 1 Questions (8 questions)
======================================
Crossing Nearby: yes (conf: 0.89)
Stairs Present: no (conf: 0.95)
Obstacle Ahead: yes (conf: 0.87)
Vehicle Nearby: yes (conf: 0.85)
...

Level 2 Follow-ups
======================================
--- Crossing details ---
  └─ Signal Present: yes (conf: 0.91)
  └─ Vehicles in Lanes: yes (conf: 0.86)

--- Obstacle details ---
  └─ Obstacle Object: yes (conf: 0.87)
  └─ Space Around: yes (conf: 0.90)

Processed 18 total questions
```

## Customization

### Adding Questions

Edit `vqa_prompts.json`:

```json
{
  "id": "q_new_question",
  "text": "Is there a new condition?",
  "type": "boolean",
  "dependency": "q_parent_question",
  "short_label": "New Question",
  "enabled": true
}
```

### Command Line Tools

```bash
# List presets
python prompt_utils.py list-presets

# Show questions
python prompt_utils.py list-questions --preset crossing

# Statistics
python prompt_utils.py stats
```
---  
**Last Updated:** November 2025
