# GPT-based Text Similarity Model

A deep learning project implementing a GPT architecture for text similarity judgment, built with PyTorch.

## 🌟 Project Highlights

- **Advanced Architecture**: Implemented GPT model with multi-head attention mechanism and transformer encoder
- **End-to-End Pipeline**: Complete training system from data preprocessing to model evaluation
- **Performance Optimization**: Achieved 0.219 loss value through architecture improvements
- **Modular Design**: Well-structured codebase with clear separation of concerns

## 🚀 Features

### Model Architecture
- Multi-head attention mechanism
- Position encoding
- Transformer encoder
- MLM (Masked Language Modeling) and NSP (Next Sentence Prediction) tasks

### Training Pipeline
- Data preprocessing and tokenization
- Model training with validation
- Performance evaluation
- Attention visualization
- Model checkpointing

### Technical Specifications
- Model dimension: 512
- Encoder layers: 6
- Attention heads: 8
- Dropout rate: 0.1
- Learning rate: 5e-5

## 📊 Performance

- Final loss value: 0.219
- Training epochs: 200
- Batch size: 16
- Dataset: MRPC (Microsoft Research Paraphrase Corpus)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt-text-similarity.git
cd gpt-text-similarity
```

2. Create and activate virtual environment:
```bash
python -m venv nlp_env
.\nlp_env\Scripts\activate  # Windows
source nlp_env/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Training the Model
```bash
python GPT.py
```

### Running the Transformer Model
```bash
python transformer.py
```

## 📁 Project Structure

```
├── GPT.py              # Main model implementation
├── transformer.py      # Transformer architecture
├── utils.py           # Data processing utilities
├── requirements.txt   # Project dependencies
└── visual/           # Visualization results
    ├── models/       # Model checkpoints
    └── tmp/         # Temporary files
```

## 🔍 Model Architecture

### Core Components
1. **Word Embedding Layer**
   - Converts input tokens to dense vectors
   - Dimension: 512

2. **Position Encoding**
   - Adds positional information to embeddings
   - Helps model understand word order

3. **Multi-head Attention**
   - 8 attention heads
   - Captures different aspects of relationships between words

4. **Transformer Encoder**
   - 6 layers of transformer blocks
   - Each layer includes attention and feed-forward networks

5. **Task Heads**
   - MLM: Predicts masked tokens
   - NSP: Judges sentence pair relationships

## 📈 Training Process

1. **Data Preparation**
   - Download MRPC dataset
   - Text preprocessing
   - Tokenization
   - Batch preparation

2. **Model Training**
   - Forward pass
   - Loss computation
   - Backward pass
   - Parameter updates

3. **Evaluation**
   - Validation loss tracking
   - Accuracy measurement
   - Attention visualization

## 🎨 Visualization

The project includes visualization tools for:
- Attention weights
- Training progress
- Model predictions

## 🔄 Future Improvements

1. **Model Enhancements**
   - Increase model size
   - Add more training data
   - Implement advanced optimization techniques

2. **Feature Additions**
   - Interactive demo
   - API interface
   - More visualization options

3. **Performance Optimization**
   - Distributed training
   - Mixed precision training
   - Model quantization

## 📚 References

1. Attention Is All You Need
2. Language Models are Few-Shot Learners
3. BERT: Pre-training of Deep Bidirectional Transformers

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- MRPC dataset providers
- All contributors to the project 