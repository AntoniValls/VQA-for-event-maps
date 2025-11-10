# VQA Pedestrian Navigation System

An improved Visual Question Answering system for pedestrian navigation and obstacle detection for people with low vision.

## 🎯 Key Improvements

### 1. **JSON-Based Prompt Management**
- All questions organized in `vqa_prompts.json`
- Easy to add, modify, or disable questions
- Configurable presets for different use cases
- Metadata tracking for each question

### 2. **Multiple Model Support**
- **ViLT**: Fast but lower accuracy (baseline)
- **BLIP-2 (2.7B)**: Good balance of speed and accuracy ⭐
- **BLIP-2 (Flan-T5-XL)**: Better accuracy, more VRAM
- **LLaVA-1.6**: State-of-the-art for safety tasks 🔥
- **InstructBLIP**: Best for complex reasoning

### 3. **Preset Configurations**
- `safety_critical`: Essential safety questions only
- `full_assessment`: Comprehensive analysis
- `environment_only`: Environmental characteristics
- `crossing_focused`: Detailed crossing assessment

### 4. **Enhanced Features**
- Better answer visualization with color coding
- Organized question categories
- Dependency tracking between questions
- Model comparison support
- Detailed logging in JSONL format

## 📁 File Structure

```
vqa_prompts.json          # Question database
vqa_improved.py           # Main processing script
prompt_utils.py           # Utility for managing prompts
README.md                 # This file
```

## 🚀 Quick Start

### Basic Usage

```python
# Run with default settings (BLIP-2, safety_critical preset)
python vqa_improved.py
```

### Configuration

Edit the configuration section in `vqa_improved.py`:

```python
MODEL = "blip2"                    # Choose your model
PROMPT_PRESET = "safety_critical"  # Choose your preset
FRAME_STRIDE = 100                 # Process every N frames
SEQUENCE = 10                      # Video sequence number
```

### Available Models

| Model | Speed | Accuracy | VRAM | Best For |
|-------|-------|----------|------|----------|
| `vilt` | ⚡⚡⚡ | ⭐ | ~4GB | Testing |
| `blip2` | ⚡⚡ | ⭐⭐⭐ | ~8GB | Production ✓ |
| `blip2-large` | ⚡ | ⭐⭐⭐⭐ | ~16GB | High accuracy |
| `llava` | ⚡ | ⭐⭐⭐⭐⭐ | ~16GB | Safety critical 🔥 |
| `instructblip` | ⚡ | ⭐⭐⭐⭐ | ~24GB | Complex reasoning |

### Available Presets

**safety_critical** (8 questions)
- Obstacle detection
- Surface hazards
- Crossing state
- Traffic signals
- Vehicle presence
- Safe to cross decision

**full_assessment** (All enabled questions)
- Complete navigation analysis
- All categories covered

**environment_only** (6 questions)
- Path width
- Surface type and hazards
- Crowding level
- Visibility conditions

**crossing_focused** (7 questions)
- Detailed crossing analysis
- Signal states
- Traffic assessment
- Curb access

## 🛠 Prompt Management

### View Available Presets

```bash
python prompt_utils.py list-presets
```

### List All Questions

```bash
# All questions
python prompt_utils.py list-questions

# Questions in a specific preset
python prompt_utils.py list-questions --preset safety_critical
```

### View Question Details

```bash
python prompt_utils.py show-question q_path_obstacle_ahead_present_5m
```

### Show Statistics

```bash
python prompt_utils.py stats
```

### Create Custom Preset

```bash
python prompt_utils.py create-preset my_preset \
  --description "Custom navigation preset" \
  --questions "q_path_obstacle_ahead_present_5m,q_crossing_approach_state,q_env_crowding_level"
```

## 📊 Question Categories

### I. Immediate Path Assessment
- Obstacle detection (5m ahead)
- Obstacle type and direction
- Navigation planning
- Overhead hazards

### II. Environment & Path Characteristics
- Path width and surface type
- Sidewalk features
- Surface hazards
- Path edge definition
- Stairs and vertical changes

### III. Street Crossing Assessment
- Crossing state and readiness
- Pedestrian signals
- Traffic analysis
- Safe crossing decisions

### IV. Surrounding Environment & Dynamic Actors
- Pedestrian crowding
- Bicycle lanes and cyclists
- Construction areas
- Visibility conditions

### V. Points of Interest
- Store entrances and accessibility
- Street signs
- Landmark identification

## 🔧 Customizing Questions

### Adding a New Question

Edit `vqa_prompts.json`:

```json
{
  "id": "q_custom_question",
  "text": "Your question text here?",
  "topic": "Topic Name",
  "type": "boolean",
  "choices": ["Yes", "No"],
  "short_label": "Custom Label",
  "enabled": true,
  "dependency": "q_other_question"  // Optional
}
```

### Question Types

- `boolean`: Yes/No questions
- `multiple_choice`: Select from predefined choices
- `bounding_box`: (Future) Object localization
- `text`: (Future) OCR or free-form answers

### Disabling a Question

Set `"enabled": false` in the question definition.

## 📈 Output Format

Results are saved in JSONL format with one line per question:

```json
{
  "frame": 100,
  "question_id": "q_path_obstacle_ahead_present_5m",
  "question": "Full question text...",
  "answer": "Yes",
  "confidence": 0.92,
  "model": "blip2"
}
```

## 🎨 Visualization

The script displays real-time results with:
- Color-coded answers (Green=safe, Red=caution, White=neutral)
- Confidence scores (when available)
- Frame information
- Model name

Press any key to advance to the next processed frame.

## 💡 Best Practices

### For Real-Time Navigation
- Use `safety_critical` preset
- Choose `blip2` for speed/accuracy balance
- Set lower `FRAME_STRIDE` (e.g., 30)

### For Comprehensive Analysis
- Use `full_assessment` preset
- Choose `llava` for best accuracy
- Higher `FRAME_STRIDE` acceptable (e.g., 100)

### For Dataset Annotation
- Use `full_assessment` preset
- Process all frames (`FRAME_STRIDE = 1`)
- Consider batch processing

## 🔍 Model Recommendations

### Budget Setup (~8GB VRAM)
```python
MODEL = "blip2"
PROMPT_PRESET = "safety_critical"
```

### Balanced Setup (~16GB VRAM)
```python
MODEL = "llava"
PROMPT_PRESET = "safety_critical"
```

### Maximum Accuracy (~24GB VRAM)
```python
MODEL = "instructblip"
PROMPT_PRESET = "full_assessment"
```

## 📝 Question Dependencies

Some questions depend on others:
- Obstacle type questions only asked if obstacle detected
- Crossing questions only asked when near crossing
- Feature hazard questions only asked if features present

This is handled automatically by the dependency system.

## 🚧 Future Enhancements

- [ ] Bounding box visualization
- [ ] OCR integration for text reading
- [ ] Spatial reasoning questions
- [ ] Multi-frame temporal reasoning
- [ ] Audio output integration
- [ ] Real-time processing mode

## 📚 Data Sources

As mentioned in the document:
- Biel Glasses dataset
- WAD Dataset (WalkVLM)
- YouTube walking channels (Roma Walking Tour, DuckTravel, etc.)
- Mapillary dataset

## 🤝 Contributing

To add new questions:
1. Add question to appropriate category in `vqa_prompts.json`
2. Test with your model of choice
3. Create custom preset if needed
4. Update documentation

## 📄 License

[Your License Here]

## ✨ Credits

Based on the original pedestrian navigation VQA system with significant improvements in organization, flexibility, and model support.
