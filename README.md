# OpenInference

OpenInference is a high-performance Python framework designed to streamline and optimize the deployment and serving of machine learning models, with a focus on Large Language Models (LLMs). It provides robust tools for model management, inference optimization, performance monitoring, and seamless integration into various environments.

## Features

* **Model Management**: Easily load, unload, and manage different versions of your ML models. Supports various model types, with a focus on transformer-based architectures including LLaMA, Mistral, Mixtral, GPT, BERT, and more.
* **Optimized Runtimes**: Integrates with hardware acceleration backends (CUDA) and optimization techniques (quantization) to deliver high-performance inference.
* **Comprehensive Monitoring**: Built-in metrics collection for latency, throughput, resource utilization, and more. Export metrics for analysis and alerting.
* **Batching Support**: Automatic request batching to improve throughput for concurrent requests, with continuous batching for LLMs.
* **LangChain Integration**: Seamless integration with LangChain for building applications on top of OpenInference models.
* **Extensible**: Designed with modularity in mind, allowing for easy extension and integration with custom components.
* **CLI Access**: Interact with the inference engine via a command-line interface.

## Project Structure

```
OpenInference/
├── api/                  # REST API for inference and management
├── batching/             # Request batching logic
├── docs/                 # Project documentation
├── hardware/             # Hardware-specific acceleration backends
├── integrations/         # Integrations with other tools and platforms
├── memory/               # Memory management utilities (e.g., KV cache)
├── models/
│   ├── converters/       # Model format conversion tools
│   └── loaders/          # Model loading utilities
├── monitoring/           # Metrics collection and monitoring
├── optimization/
│   ├── graph/            # Graph optimization techniques
│   ├── kernels/          # Custom optimized kernels
│   └── quantization/     # Model quantization tools
├── runtime/
│   ├── core/             # Core inference runtime logic
│   ├── cuda/             # CUDA-specific runtime components
│   └── metal/            # Metal-specific runtime components
├── tests/                # Unit and integration tests
├── ui/                   # Web UI for monitoring and management
├── __init__.py
├── .env                  # Environment configuration (template as .env.example)
├── cli.py                # Command-line interface
├── main.py               # Main application entry point and OpenInference server
├── setup.py              # Packaging and installation script
└── LICENSE               # Project License
```

## Getting Started

### Prerequisites

* Python 3.11+
* Pip (Python package installer)
* (Optional but Recommended) NVIDIA GPU with CUDA drivers for GPU acceleration

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/llamasearchai/OpenInference.git
   cd OpenInference
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install OpenInference in editable mode
   ```
   *Note: `requirements.txt` provides a list of core dependencies. For development, also install `requirements-dev.txt` as shown below.*

   For development, including tools for testing, linting, and documentation, install:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **(Optional) Create a `.env` file for environment-specific configurations:**
   You can copy `.env.example` to `.env` and customize it.
   Example `.env` content:
   ```env
   LOG_LEVEL=INFO
   METRICS_EXPORT_PATH=/tmp/openinference_metrics
   # HUGGING_FACE_HUB_TOKEN=your_hf_token_here  # If needed for private models
   ```

### Running Tests

To ensure everything is set up correctly and the core functionalities are working, run the tests:

```bash
python -m unittest discover -s tests
```

## Usage

OpenInference provides a Command Line Interface (`cli.py`) for managing and interacting with the inference server.

### CLI Examples

* **List available models:**
  ```bash
  python cli.py list-models
  ```

* **Get hardware information:**
  ```bash
  python cli.py hardware-info
  ```

* **Start the inference server:**
  ```bash
  python cli.py serve --host 0.0.0.0 --port 8000
  ```

* **Benchmark a model:**
  ```bash
  python cli.py benchmark model_name --batch-sizes 1 4 8 --sequence-lengths 32 128 512
  ```

* **Quantize a model:**
  ```bash
  python cli.py quantize model_name --precision int8
  ```

### Programmatic Usage

```python
from openinference import OpenInference

# Initialize the inference engine
engine = OpenInference(device="cuda")

# Load a model
engine.load_model("mistral-7b-instruct")

# Run inference
response = engine.run_inference(
    model_name="mistral-7b-instruct",
    inputs="What is the capital of France?",
    max_new_tokens=128
)

# For streaming responses
for chunk in engine.run_inference(
    model_name="mistral-7b-instruct",
    inputs="Explain quantum computing",
    stream=True,
    max_new_tokens=512
):
    print(chunk, end="", flush=True)

# Using with LangChain
from integrations.langchain_adapter import OpenInferenceLLM

llm = OpenInferenceLLM(
    model_name="mistral-7b-instruct",
    inference_engine=engine,
    temperature=0.7
)

result = llm("What are the main features of Python?")
```

### Monitoring

Metrics are collected automatically and can be exported to a specified path (see `METRICS_EXPORT_PATH` in `.env`).

## Contributing

We welcome contributions to OpenInference! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute, report issues, and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 