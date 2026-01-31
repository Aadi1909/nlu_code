# Battery Smart Voice Assistant - NLU Model

Intent classification and entity extraction for Battery Smart's multilingual voice assistant (Hindi, English, Hinglish).

## 🎯 Features

- **23 Intent Classes**: Battery status, swap stations, payments, subscriptions, etc.
- **Multilingual Support**: Hindi, English, and Hinglish
- **MuRIL Model**: Google's Multilingual Representations for Indian Languages
- **Augmented Dataset**: 1,230+ training examples
- **High Performance**: 70%+ accuracy on test set

## 📊 Model Performance

### Intent Classifier
- **Model**: `google/muril-base-cased`
- **Training Examples**: 1,230 (augmented from 342)
- **Test Accuracy**: 70%+ (improved from 48.6%)
- **Supported Intents**: 23 classes

### Dataset Statistics
```
Training: 1,230 examples
Validation: 153 examples
Test: 155 examples
Languages: Hindi, English, Hinglish
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd model_proj

# Install dependencies
pip install -r requirements.txt

# Configure Python environment (if using venv)
python -m venv nlu_env
source nlu_env/bin/activate  # On Windows: nlu_env\Scripts\activate
pip install -r requirements.txt
```

### Training

```bash
# Run data augmentation
cd src
python3 improve_and_retrain.py

# Train intent classifier
python3 train_intent_classifier.py
```

### Testing

```bash
# Interactive testing
cd src
python3 test_intent_model.py
```

## 📁 Project Structure

```
model_proj/
├── config/              # Intent, entity, domain configurations
│   ├── intents.yaml
│   ├── entities.yaml
│   └── domain.yaml
├── data/
│   ├── processed/       # Training data
│   │   ├── train_augmented.json
│   │   ├── val_augmented.json
│   │   └── test_augmented.json
│   └── raw/            # Original data
├── models/             # Trained models
│   └── intent_classifier/
├── src/
│   ├── train_intent_classifier.py  # Training script
│   ├── test_intent_model.py        # Testing script
│   ├── improve_and_retrain.py      # Data augmentation
│   ├── training/                    # Training utilities
│   ├── nlu/                        # NLU pipeline
│   └── api/                        # FastAPI endpoints
└── requirements.txt
```

## 🎓 Supported Intents

1. **Battery & Charging**
   - `check_battery_status` - Current charge level
   - `check_battery_range` - Remaining range
   
2. **Swap Stations**
   - `find_swap_station` - Locate nearest station
   - `check_station_availability` - Station status
   - `station_directions` - Get directions

3. **Subscriptions & Payments**
   - `check_subscription` - Plan details
   - `subscription_renewal` - Renew plan
   - `upgrade_subscription` - Upgrade plan
   - `make_payment` - Process payment
   - `payment_inquiry` - Payment status
   - `payment_history` - Transaction history

4. **Usage & History**
   - `swap_history` - Past swaps
   - `swap_count` - Total swaps
   - `check_ride_stats` - Ride statistics

5. **Support**
   - `report_issue` - Report problems
   - `talk_to_agent` - Human support
   - `help_general` - General help

6. **Conversation**
   - `greet`, `bye`, `thank`, `affirm`, `deny`

## 🔧 Configuration

### Training Hyperparameters
```python
num_epochs: 15
batch_size: 16
learning_rate: 3e-5
early_stopping_patience: 5
```

### Data Augmentation
- Hinglish variations
- Polite form additions
- Class balancing (30+ samples per class)
- Text transformations

## 📈 Improvement Roadmap

- [x] Data augmentation
- [x] Balanced dataset
- [x] Improved hyperparameters
- [ ] Entity extraction training
- [ ] Complete NLU pipeline
- [ ] API deployment
- [ ] Multi-turn dialogue support

## 🤝 Contributing

See `TRAINING_GUIDE.md` for details on training on AWS Bedrock and contributing improvements.

## 📄 License

[Your License Here]

## 📞 Contact

[Your Contact Information]
